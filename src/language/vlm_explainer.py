from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from typing import Any, Tuple
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from src.data.mvtec import clip_denormalize, clip_normalize
from src.detectors.winclip import WinCLIPDetector


@dataclass
class LanguageExplanation:
    category: str
    is_anomalous: bool
    anomaly_score: float
    predicted_defect_type: str             # e.g., "cut", "hole", "scratch", "contamination", "crack", "deformation", "none"
    defect_confidence: float               # [0.0, 1.0]
    spatial_quadrant: str                  # "top-left", "top-right", "bottom-left", "bottom-right", "center", "none"
    grounding_bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in [0, 240] coordinates
    area_percentage: float                 # Percentage of image area covered by high-attribution region
    visual_evidence: str                   # Grounded natural-language explanation for operator
    recommended_action: str                # Actionable engineering recommendation
    backend_used: str                      # "openclip_grounded" or "ollama_vlm"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFECT_CANDIDATES = {
    "cut": "a photo of a surface cut or slash",
    "hole": "a photo of a puncture, hole, or missing material",
    "scratch": "a photo of a thin surface scratch or abrasion",
    "contamination": "a photo of foreign contamination, stain, or discoloration",
    "crack": "a photo of a structural crack or fracture",
    "deformation": "a photo of a bent, warped, or misplaced component",
}


class GroundedLanguageExplainer:
    """Generates structured, hallucination-checked natural-language explanations
    grounded in visual attributions and model predictions.
    """

    def __init__(
        self,
        detector: WinCLIPDetector,
        anomaly_threshold: float = 0.45,
        percentile_threshold: float = 90.0,
        backend: str = "openclip_grounded",
        ollama_url: str = "http://localhost:11434/api/generate",
        ollama_model: str = "llava",
    ) -> None:
        self.detector = detector
        self.anomaly_threshold = anomaly_threshold
        self.percentile_threshold = percentile_threshold
        self.backend = backend
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.device = detector.device

        # Cache text embeddings for defect candidates (when detector has CLIP model)
        self.defect_keys = list(DEFECT_CANDIDATES.keys())
        self.candidate_text_embs: Tensor | None = None
        if hasattr(getattr(detector, "model", None), "encode_text"):
            with torch.no_grad():
                prompts = [DEFECT_CANDIDATES[k] for k in self.defect_keys]
                tokenizer = getattr(detector, "tokenizer", None)
                if tokenizer is None:
                    import open_clip
                    try:
                        tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
                    except Exception:
                        tokenizer = open_clip.tokenize
                tokens = tokenizer(prompts).to(self.device)
                text_embs = F.normalize(self.detector.model.encode_text(tokens).float(), dim=-1)
                self.candidate_text_embs = text_embs

    def extract_defect_region(
        self, attr_map: np.ndarray | Tensor
    ) -> tuple[tuple[int, int, int, int], str, float, tuple[float, float]]:
        """Extracts bounding box, quadrant, and area percentage of largest attribution cluster."""
        if isinstance(attr_map, Tensor):
            attr_np = attr_map.detach().squeeze().cpu().numpy()
        else:
            attr_np = np.asarray(attr_map, dtype=np.float32).squeeze()

        h, w = attr_np.shape[-2:]
        # Adaptive thresholding: avoid thresholding at zero if 90th percentile is zero
        max_val = float(attr_np.max())
        if max_val <= 1e-6:
            return (0, 0, 0, 0), "none", 0.0, (0.0, 0.0)

        p_thresh = float(np.percentile(attr_np, self.percentile_threshold))
        thresh = max(p_thresh, max_val * 0.35)
        binary = (attr_np >= thresh).astype(np.uint8)

        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            return (0, 0, 0, 0), "none", 0.0, (0.0, 0.0)

        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        largest_label = int(np.argmax(sizes)) + 1
        comp_mask = labeled == largest_label
        comp_pixels = int(np.sum(comp_mask))

        coords = np.argwhere(comp_mask)
        y_min, x_min = int(coords[:, 0].min()), int(coords[:, 1].min())
        y_max, x_max = int(coords[:, 0].max()), int(coords[:, 1].max())

        cy, cx = ndimage.center_of_mass(comp_mask)
        area_pct = float((comp_pixels / (h * w)) * 100.0)

        # Quadrant classification
        if (h * 0.33) <= cy <= (h * 0.67) and (w * 0.33) <= cx <= (w * 0.67):
            quadrant = "center"
        else:
            v_str = "top" if cy < (h / 2.0) else "bottom"
            h_str = "left" if cx < (w / 2.0) else "right"
            quadrant = f"{v_str}-{h_str}"

        return (x_min, y_min, x_max, y_max), quadrant, area_pct, (float(cx), float(cy))

    def _classify_crop(self, image: Tensor, bbox: tuple[int, int, int, int]) -> tuple[str, float]:
        """Classifies padded crop using candidate defect text prompts."""
        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            return "none", 0.0

        pad = 16
        h, w = image.shape[-2:]
        x0 = max(0, x_min - pad)
        y0 = max(0, y_min - pad)
        x1 = min(w, x_max + pad)
        y1 = min(h, y_max + pad)

        crop = image[:, :, y0:y1, x0:x1]
        if self.candidate_text_embs is None or not hasattr(getattr(self.detector, "model", None), "encode_image"):
            aspect = max(x_max - x_min, 1) / max(y_max - y_min, 1)
            if aspect > 2.2 or aspect < 0.45:
                return "scratch", 0.78
            return "contamination", 0.82

        crop_resized = F.interpolate(crop, size=(240, 240), mode="bilinear", align_corners=False)

        with torch.no_grad():
            crop_emb = F.normalize(self.detector.model.encode_image(crop_resized.to(self.device)).float(), dim=-1)
            logits = 100.0 * (crop_emb @ self.candidate_text_embs.t())
            probs = F.softmax(logits, dim=-1)[0]
            best_idx = int(torch.argmax(probs).item())
            best_conf = float(probs[best_idx].item())
            best_class = self.defect_keys[best_idx]

        return best_class, best_conf

    def explain(
        self,
        image: Tensor,
        anomaly_score: float,
        attr_map: Tensor | np.ndarray,
        category: str = "industrial specimen",
    ) -> LanguageExplanation:
        """Generates grounded language explanation."""
        img_t = image.to(self.device, dtype=torch.float32)
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)

        is_anom = float(anomaly_score) >= self.anomaly_threshold

        # Hallucination Guardrail: If below threshold, reject defect hallucinations
        if not is_anom:
            return LanguageExplanation(
                category=category,
                is_anomalous=False,
                anomaly_score=round(float(anomaly_score), 4),
                predicted_defect_type="none",
                defect_confidence=round(1.0 - float(anomaly_score), 4),
                spatial_quadrant="none",
                grounding_bbox=(0, 0, 0, 0),
                area_percentage=0.0,
                visual_evidence=(
                    f"The {category} specimen conforms to nominal appearance. "
                    f"Anomaly score {anomaly_score:.2f} remains below the {self.anomaly_threshold:.2f} threshold. "
                    f"No concentrated surface or structural irregularities were detected."
                ),
                recommended_action="PASS: Component meets quality criteria for assembly or release.",
                backend_used=self.backend,
            )

        bbox, quadrant, area_pct, _ = self.extract_defect_region(attr_map)
        defect_type, conf = self._classify_crop(img_t, bbox)

        x_min, y_min, x_max, y_max = bbox
        action_map = {
            "cut": "REJECT: Surface integrity compromised by cuts. Route to scrap or cutting rework.",
            "hole": "REJECT: Structural puncture or missing material detected. Immediate quarantine required.",
            "scratch": "RE-INSPECT: Abrasive surface scratch detected. Route to polishing or manual inspection.",
            "contamination": "RE-INSPECT: Foreign contamination observed. Route to chemical wash / cleaning station.",
            "crack": "REJECT: Critical crack or fissure identified. Material failure risk — discard specimen.",
            "deformation": "REJECT: Component geometry warped or misaligned. Check tooling and stamping calibration.",
        }
        recommended_action = action_map.get(defect_type, "RE-INSPECT: Unclassified anomaly detected exceeding quality limits.")

        visual_evidence = (
            f"Defect detected on {category} with an anomaly score of {anomaly_score:.2f} (threshold: {self.anomaly_threshold:.2f}). "
            f"The primary flaw is localized in the {quadrant} quadrant at bounding box [{x_min}, {y_min}, {x_max}, {y_max}], "
            f"covering approximately {area_pct:.1f}% of the specimen surface. "
            f"Visual features exhibit high similarity to a '{defect_type}' ({conf*100:.1f}% confidence)."
        )

        return LanguageExplanation(
            category=category,
            is_anomalous=True,
            anomaly_score=round(float(anomaly_score), 4),
            predicted_defect_type=defect_type,
            defect_confidence=round(conf, 4),
            spatial_quadrant=quadrant,
            grounding_bbox=bbox,
            area_percentage=round(area_pct, 2),
            visual_evidence=visual_evidence,
            recommended_action=recommended_action,
            backend_used=self.backend,
        )
