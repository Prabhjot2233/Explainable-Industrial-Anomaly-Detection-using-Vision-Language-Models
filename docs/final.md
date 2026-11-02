# COMP 8977 — All-Out Master Implementation Plan (Weeks 5–8 + Full-Stack Frontend & Final Demo)

**Project:** Explainable Industrial Anomaly Detection using Vision-Language Models (Team 12: Prabhjot Singh & Sarhan Kapadiya)  
**Target Agent:** Antigravity Autonomous Coding Agent  
**Prerequisite Status:** Weeks 1–4 (MVTec Loader, Detection Metrics, WinCLIP Zero-Shot/Few-Shot, AnomalyCLIP, PatchCore, Differentiable Score Wrapper, Saliency, Integrated Gradients, ViT Grad-CAM, and Visualization Tools) are **COMPLETE**.  
**Mission:** Complete the entire remaining project (Weeks 5, 6, 7, and 8) including **ABIGX (AFR + IG)**, **Quantitative XAI Metrics (AUC, SUM, ADD, DEL)**, **Grounded VLM Language Explanations**, **Language Hallucination & Grounding Metrics**, **IEEE Report Table Generators**, and a **Polished Multi-Tab Interactive Web Frontend (`app/streamlit_app.py`)** ready for next week's live demo.

---

## 0. CRITICAL DIRECTIVES FOR THE ANTIGRAVITY AGENT

1. **Pre-Flight Check (Verify Weeks 1–4 First):**
   * Before starting Step 18, run `pytest -v` to confirm all Weeks 1–4 tests pass.
   * Do **NOT** break any existing interfaces in `src/data/mvtec.py`, `src/detectors/`, or `src/explainers/`.

2. **Sequential Step-by-Step Execution (Steps 18 to 24):**
   * Build strictly in the order defined in **Section 3** (Step 18 $\rightarrow$ Step 24), **one step at a time**.
   * **STOP** after completing each step, run its exact `pytest` or CLI acceptance command, update the `weekly_progress/` beginner guide for that step, and show the terminal output to the user before moving on.

3. **Pure PyTorch Differentiable Forward Path:**
   * Keep all operations inside `ABIGXExplainer` (Adversarial Fault Reconstruction PGD loop + Integrated Gradients) strictly in PyTorch (`torch.float32`) using `DifferentiableScoreWrapper(detector, score_mode="combined_smooth")`.
   * **NO NumPy or OpenCV (`cv2`)** inside any gradient loop.

4. **Cross-Platform Mandate (macOS MPS & Windows CUDA/CPU):**
   * Use `pathlib.Path` everywhere and `.as_posix()` when serializing paths to JSON/CSV.
   * Always specify `encoding="utf-8"` when reading or writing YAML, CSV, JSON, LaTeX, or Markdown files.
   * Default `num_workers: 0` in all DataLoaders for Windows (`spawn`) and macOS (`mps`) stability.
   * Use `get_device()` from `src/utils/device.py` (`cuda` $\rightarrow$ `mps` $\rightarrow$ `cpu`).

5. **Beginner-Friendly Weekly Tracking (`weekly_progress/`):**
   * For every step below, update the corresponding guide inside `weekly_progress/` so someone brand new to Python and Machine Learning can understand:
     * **What We Built & Why It Matters** (using real-world factory analogies)
     * **How the Math Works Step-by-Step**
     * **How the Code Works**
     * **Exact Commands to Run & What Output to Expect**

---

## 1. COMPLETE REPOSITORY LAYOUT (WEEKS 5–8 ADDITIONS)

Add the following files to the existing repository structure:

```text
.
├── app/
│   ├── __init__.py
│   └── streamlit_app.py               # Step 23: Full-Stack 4-Tab Interactive Web Frontend
├── configs/
│   ├── explanations.yaml              # Step 20: Config for ABIGX & XAI benchmark
│   └── language.yaml                  # Step 21: Config for Grounded Language Explainer
├── scripts/
│   ├── run_explanations.py            # Step 20: Benchmark all 5 visual explainers + AFR ablation
│   ├── run_language_eval.py           # Step 22: Evaluate language accuracy, grounding & hallucination
│   └── generate_report_artifacts.py   # Step 24: Auto-generate IEEE LaTeX/Markdown tables & plots
├── src/
│   ├── explainers/
│   │   ├── abigx.py                   # Step 18: ABIGX (Adversarial Fault Reconstruction + IG)
│   │   └── visualize.py               # Update: Add AFR trajectory & DEL/ADD curve plotting
│   ├── language/
│   │   ├── __init__.py
│   │   └── vlm_explainer.py           # Step 21: Grounded VLM + OpenCLIP Structured Text Explainer
│   └── metrics/
│       ├── explanation.py             # Step 19: Correctness (AUC, SUM) & Consistency (DEL, ADD)
│       └── language_eval.py           # Step 22: Defect Accuracy, Quadrant Match, IoU, Hallucination Rate
├── tests/
│   ├── test_abigx.py                  # Step 18 acceptance test
│   ├── test_explanation_metrics.py    # Step 19 acceptance test
│   ├── test_language_explainer.py     # Step 21 acceptance test
│   ├── test_language_metrics.py       # Step 22 acceptance test
│   └── test_frontend_smoke.py         # Step 23 acceptance test
└── weekly_progress/
    ├── week5_abigx_and_xai_metrics.md
    ├── week6_benchmark_and_vlm_text.md
    ├── week7_language_eval_and_frontend.md
    └── week8_demo_playbook_and_ieee_report.md
```

---

## 2. FIXED INTERFACES & MATHEMATICAL CONTRACTS (WEEKS 5–8)

### 2.1 ABIGX Explainer (`src/explainers/abigx.py`)
ABIGX (Zhuo et al., 2026) solves the baseline problem of Integrated Gradients in two stages:

1. **Stage 1 — Adversarial Fault Reconstruction (AFR):**
   Given an anomalous input image $x^{(0)} \in \mathbb{R}^{1 \times 3 \times 240 \times 240}$, find a minimally modified counter-factual baseline image $x_{\text{AFR}}$ inside an $L_\infty$ ball of radius $\epsilon$ around $x^{(0)}$ (clamped to valid CLIP-normalized pixel bounds) that minimizes the smooth anomaly score plus an $L_2$ proximity penalty:
   $$\mathcal{L}_{\text{AFR}}\left(x^{(t)}\right) = F_{\text{smooth}}\left(x^{(t)}\right) + \lambda_{\text{prox}} \left\| x^{(t)} - x^{(0)} \right\|_2^2$$
   Using Projected Gradient Descent (PGD) for $t = 0, \dots, T-1$:
   $$g^{(t)} = \nabla_{x^{(t)}} \mathcal{L}_{\text{AFR}}\left(x^{(t)}\right)$$
   $$x^{(t+1)} = \Pi_{\text{valid}}\left( \Pi_{\mathcal{B}_\epsilon\left(x^{(0)}\right)}\left( x^{(t)} - \alpha \cdot \operatorname{sign}\left(g^{(t)}\right) \right) \right)$$
   * **Early Stopping:** Stop as soon as $F_{\text{smooth}}\left(x^{(t)}\right) \le \tau_{\text{normal}}$ (default `0.35`).
   * **Convergence Check & Fallback (Proposal Section 4.4 Mitigation):**
     Define relative score drop:
     $$\Delta_{\text{rel}} = \frac{F_{\text{smooth}}\left(x^{(0)}\right) - F_{\text{smooth}}\left(x^{(T)}\right)}{\max\left(F_{\text{smooth}}\left(x^{(0)}\right), 10^{-6}\right)}$$
     If $F_{\text{smooth}}\left(x^{(T)}\right) \le \tau_{\text{normal}}$ or $\Delta_{\text{rel}} \ge 0.15$, mark `converged = True` and integrate gradients from $x_{\text{AFR}} = x^{(T)}$ to $x^{(0)}$.
     Otherwise, mark `converged = False` and **automatically fall back** to Expected Integrated Gradients using `reference_normals` (or a zero baseline if `reference_normals` is `None`).

2. **Stage 2 — Path Integration (IG from $x_{\text{AFR}}$ to $x^{(0)}$):**
   $$\text{IG}_i\left(x^{(0)}, x_{\text{AFR}}\right) = \left(x_i^{(0)} - x_{\text{AFR}, i}\right) \times \frac{1}{M} \sum_{m=1}^{M} \frac{\partial F_{\text{smooth}}\left(x_{\text{AFR}} + \frac{m}{M}\left(x^{(0)} - x_{\text{AFR}}\right)\right)}{\partial x_i}$$
   Aggregate across RGB channels via L2 norm (`torch.linalg.vector_norm(raw_attr, ord=2, dim=1)`) and min-max normalize to `[0, 1]` of shape `[1, 240, 240]`.

---

### 2.2 Quantitative Explanation Metrics (`src/metrics/explanation.py`)

All four metrics must be implemented cleanly and accept `[H, W]` NumPy arrays (for Correctness) or PyTorch Tensors + Detector (for Consistency):

1. **`correctness_auc(attr_map: np.ndarray, gt_mask: np.ndarray) -> float`**
   * Computes `roc_auc_score(gt_mask.astype(int).ravel(), attr_map.astype(np.float32).ravel())`.
   * Returns `float("nan")` if `gt_mask` has only 0s or only 1s; returns `0.5` if `attr_map` is constant.

2. **`correctness_sum(attr_map: np.ndarray, gt_mask: np.ndarray) -> float`**
   * Measures concentration of attribution mass inside the true defect mask:
     $$\text{SUM} = \frac{\sum_{(i,j)} \left|A(i,j)\right| \cdot \mathbb{I}\left[\text{mask}(i,j) = 1\right]}{\sum_{(i,j)} \left|A(i,j)\right| + 10^{-8}}$$
   * Returns `0.0` if total attribution mass is `0.0` or `gt_mask` has no anomalous pixels.

3. **`consistency_del(detector: BaseDetector, image: Tensor, attr_map: Tensor, baseline_image: Tensor | None = None, steps: int = 10) -> dict[str, Any]`**
   * Evaluates **Deletion Faithfulness** under `with torch.no_grad():`.
   * Ranks all $H \times W$ pixels from highest attribution to lowest attribution.
   * For fraction $p \in \text{linspace}(0.0, 1.0, \text{steps} + 1)$:
     Replaces the top $p$ fraction of pixels in `image` with `baseline_image` (defaulting to `torch.zeros_like(image)` if `None`), runs `score, _ = detector(modified_image)`, and records the score.
   * Normalizes the deletion curve by dividing by ` max(initial_score, 1e-6)` (or raw area if `initial_score` is already in `[0, 1]`) and integrates using the trapezoid rule (`np.trapezoid(scores, fractions)`).
   * Returns `{"audc": float, "fractions": list[float], "scores": list[float]}`. (**Lower `audc` = better faithfulness** because removing important defect pixels causes the anomaly score to drop rapidly).

4. **`consistency_add(detector: BaseDetector, normal_image: Tensor, anomalous_image: Tensor, attr_map: Tensor, steps: int = 10) -> dict[str, Any]`**
   * Evaluates **Addition / Insertion Faithfulness** under `with torch.no_grad():`.
   * Starting from `normal_image`, progressively pastes the top $p$ fraction of highest-attributed pixels from `anomalous_image` into `normal_image`.
   * Records `score, _ = detector(composite_image)` at each step and integrates via trapezoid rule (`np.trapezoid(scores, fractions)`).
   * Returns `{"auac": float, "fractions": list[float], "scores": list[float]}`. (**Higher `auac` = better faithfulness** because inserting real defect pixels into a normal image causes the anomaly score to rise rapidly).

---

### 2.3 Grounded Language Explanation Contract (`src/language/vlm_explainer.py`)

```python
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class LanguageExplanation:
    category: str
    is_anomalous: bool
    anomaly_score: float
    predicted_defect_type: str             # e.g., "cut", "hole", "scratch", "contamination", "crack", "misalignment", "none"
    defect_confidence: float               # [0.0, 1.0]
    spatial_quadrant: str                  # "top-left", "top-right", "bottom-left", "bottom-right", "center", "none"
    grounding_bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in [0, 240] coordinates
    area_percentage: float                 # Percentage of image area covered by high-attribution region
    visual_evidence: str                   # 2-3 sentence grounded natural-language explanation for operator
    recommended_action: str                # Actionable engineering recommendation
    backend_used: str                      # "openclip_grounded" or "vlm_api"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

---

## 3. STEP-BY-STEP BUILD PLAN & REFERENCE CODE (STEPS 18–24)

Execute these 7 steps in strict sequential order.

---

### STEP 18 (Week 5A): ABIGX-Style Explainer (AFR Baseline Search + IG)

**Goal:** Implement `ABIGXExplainer` in `src/explainers/abigx.py` with PGD baseline reconstruction, convergence monitoring, automatic Expected-IG fallback, and trajectory logging for the frontend demo.

#### File to Create: `src/explainers/abigx.py`
```python
from typing import Any
import torch
from torch import Tensor
from captum.attr import IntegratedGradients

from src.detectors.base import BaseDetector
from src.explainers.base import BaseExplainer, DifferentiableScoreWrapper

# CLIP normalized bounds for RGB in [0, 1]
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ABIGXExplainer(BaseExplainer):
    """ABIGX-Style Explainer: Adversarial Fault Reconstruction (AFR) + Integrated Gradients.

    Stage 1 (AFR): Uses Projected Gradient Descent (PGD) to reconstruct a normal-looking
                   counterfactual baseline x_AFR from the anomalous input x.
    Stage 2 (IG):  Accumulates gradients along the linear path from x_AFR to x.
    Fallback:      If AFR fails to reduce the anomaly score sufficiently, falls back
                   to Expected-IG over normal reference images (or zero baseline).
    """

    def __init__(
        self,
        detector: BaseDetector,
        device: torch.device,
        afr_steps: int = 15,
        afr_step_size: float = 0.03,
        afr_eps: float = 0.35,
        afr_lambda_prox: float = 0.05,
        normal_score_threshold: float = 0.35,
        min_relative_drop: float = 0.15,
        ig_steps: int = 25,
        score_mode: str = "combined_smooth",
    ) -> None:
        super().__init__(detector=detector, device=device)
        self.afr_steps = afr_steps
        self.afr_step_size = afr_step_size
        self.afr_eps = afr_eps
        self.afr_lambda_prox = afr_lambda_prox
        self.normal_score_threshold = normal_score_threshold
        self.min_relative_drop = min_relative_drop
        self.ig_steps = ig_steps
        self.wrapper = DifferentiableScoreWrapper(detector, score_mode=score_mode).to(device)
        self.ig = IntegratedGradients(self.wrapper)

        mean = torch.tensor(CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.min_val = (0.0 - mean) / std
        self.max_val = (1.0 - mean) / std

    def _run_afr(self, x0: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """Runs PGD to find a normal counterfactual baseline x_AFR near x0."""
        x_curr = x0.clone().detach()
        trajectory_scores: list[float] = []

        with torch.no_grad():
            init_score = float(self.wrapper(x0).item())
        trajectory_scores.append(init_score)

        best_x = x_curr.clone()
        best_score = init_score

        for _ in range(self.afr_steps):
            x_curr = x_curr.detach().requires_grad_(True)
            score = self.wrapper(x_curr).sum()
            prox_loss = torch.mean((x_curr - x0) ** 2)
            total_loss = score + self.afr_lambda_prox * prox_loss

            grad = torch.autograd.grad(total_loss, x_curr, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                x_next = x_curr - self.afr_step_size * torch.sign(grad)
                # Project onto L_inf epsilon-ball around x0
                delta = torch.clamp(x_next - x0, min=-self.afr_eps, max=self.afr_eps)
                x_curr = torch.clamp(x0 + delta, min=self.min_val, max=self.max_val)

                curr_score = float(self.wrapper(x_curr).item())
                trajectory_scores.append(curr_score)

                if curr_score < best_score:
                    best_score = curr_score
                    best_x = x_curr.clone()

                if curr_score <= self.normal_score_threshold:
                    break

        rel_drop = (init_score - best_score) / max(abs(init_score), 1e-6)
        converged = (best_score <= self.normal_score_threshold) or (rel_drop >= self.min_relative_drop)

        meta = {
            "initial_score": init_score,
            "baseline_score": best_score,
            "relative_drop": float(rel_drop),
            "converged": bool(converged),
            "steps_run": len(trajectory_scores) - 1,
            "trajectory_scores": trajectory_scores,
        }
        return best_x.detach(), meta

    def explain(
        self,
        image: Tensor,
        reference_normals: Tensor | None = None,
    ) -> dict[str, Any]:
        """Computes ABIGX attribution map [1, H, W] and returns baseline + convergence metadata."""
        x0 = image.to(self.device, dtype=torch.float32)
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        afr_baseline, meta = self._run_afr(x0)

        if meta["converged"]:
            meta["baseline_type"] = "afr_reconstructed"
            raw_attr = self.ig.attribute(
                x0,
                baselines=afr_baseline,
                n_steps=self.ig_steps,
                internal_batch_size=1,
            )
            used_baseline = afr_baseline
        else:
            # Proposal Section 4.4 Mitigation: Fallback to Expected-IG on normal baselines
            if reference_normals is not None and reference_normals.numel() > 0:
                meta["baseline_type"] = "fallback_expected_ig"
                refs = reference_normals.to(self.device, dtype=torch.float32)
                attrs = []
                for k in range(refs.shape[0]):
                    base_k = refs[k : k + 1]
                    attr_k = self.ig.attribute(
                        x0,
                        baselines=base_k,
                        n_steps=max(10, self.ig_steps // 2),
                        internal_batch_size=1,
                    )
                    attrs.append(attr_k)
                raw_attr = torch.stack(attrs, dim=0).mean(dim=0)
                used_baseline = refs[0:1]
            else:
                meta["baseline_type"] = "fallback_black_ig"
                used_baseline = torch.zeros_like(x0)
                raw_attr = self.ig.attribute(
                    x0,
                    baselines=used_baseline,
                    n_steps=self.ig_steps,
                    internal_batch_size=1,
                )

        # Channel aggregation via L2 norm -> [1, H, W]
        spatial_map = torch.linalg.vector_norm(raw_attr, ord=2, dim=1)
        b_min = spatial_map.amin(dim=(1, 2), keepdim=True)
        b_max = spatial_map.amax(dim=(1, 2), keepdim=True)
        norm_map = (spatial_map - b_min) / (b_max - b_min + 1e-8)

        return {
            "attribution_map": norm_map.detach(),
            "raw_attribution": raw_attr.detach(),
            "baseline": used_baseline.detach(),
            "meta": meta,
        }
```

#### File to Create: `tests/test_abigx.py`
* Test 1: Run `ABIGXExplainer` on a dummy anomalous tensor `[1, 3, 240, 240]` with `afr_steps=3` and `ig_steps=4`. Assert `attribution_map` has shape `[1, 240, 240]`, values in `[0.0, 1.0]`, and `meta["baseline_score"] <= meta["initial_score"]`.
* Test 2: Force non-convergence (`min_relative_drop=0.999`, `normal_score_threshold=-1.0`), pass `reference_normals` of shape `[2, 3, 240, 240]`, and assert `meta["converged"] is False` and `meta["baseline_type"] == "fallback_expected_ig"`.
* **Acceptance Command:** `pytest tests/test_abigx.py -v` and update `weekly_progress/week5_abigx_and_xai_metrics.md`.

---

### STEP 19 (Week 5B): Quantitative Explanation Metrics (Correctness & Consistency)

**Goal:** Implement `correctness_auc`, `correctness_sum`, `consistency_del`, and `consistency_add` in `src/metrics/explanation.py` plus unit tests in `tests/test_explanation_metrics.py`.

#### File to Create: `src/metrics/explanation.py`
```python
from typing import Any
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor

from src.detectors.base import BaseDetector


def correctness_auc(attr_map: np.ndarray | Tensor, gt_mask: np.ndarray | Tensor) -> float:
    """Computes pixel-level ROC-AUC between an attribution map and ground-truth defect mask."""
    if isinstance(attr_map, Tensor):
        attr_map = attr_map.detach().cpu().numpy()
    if isinstance(gt_mask, Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    flat_attr = np.asarray(attr_map, dtype=np.float32).ravel()
    flat_mask = (np.asarray(gt_mask, dtype=np.float32).ravel() > 0.5).astype(np.int32)

    if len(np.unique(flat_mask)) < 2:
        return float("nan")
    if np.allclose(flat_attr, flat_attr[0]):
        return 0.5

    return float(roc_auc_score(flat_mask, flat_attr))


def correctness_sum(attr_map: np.ndarray | Tensor, gt_mask: np.ndarray | Tensor) -> float:
    """Computes the fraction of total attribution mass falling inside the ground-truth mask."""
    if isinstance(attr_map, Tensor):
        attr_map = attr_map.detach().cpu().numpy()
    if isinstance(gt_mask, Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    attr = np.abs(np.asarray(attr_map, dtype=np.float32)).squeeze()
    mask = (np.asarray(gt_mask, dtype=np.float32).squeeze() > 0.5).astype(np.float32)

    total_mass = float(np.sum(attr))
    if total_mass <= 1e-8 or float(np.sum(mask)) == 0.0:
        return 0.0

    inside_mass = float(np.sum(attr * mask))
    return float(np.clip(inside_mass / (total_mass + 1e-8), 0.0, 1.0))


def consistency_del(
    detector: BaseDetector,
    image: Tensor,
    attr_map: Tensor | np.ndarray,
    baseline_image: Tensor | None = None,
    steps: int = 10,
) -> dict[str, Any]:
    """Deletion test: progressively replaces top-attributed pixels with baseline values.
    Lower AUDC (Area Under Deletion Curve) indicates a more faithful explanation.
    """
    device = next(detector.parameters()).device if list(detector.parameters()) else image.device
    x = image.to(device, dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0)

    if baseline_image is None:
        base = torch.zeros_like(x)
    else:
        base = baseline_image.to(device, dtype=torch.float32)
        if base.ndim == 3:
            base = base.unsqueeze(0)

    if isinstance(attr_map, np.ndarray):
        attr_t = torch.from_numpy(attr_map).to(device, dtype=torch.float32)
    else:
        attr_t = attr_map.to(device, dtype=torch.float32)
    attr_flat = attr_t.reshape(-1)
    sorted_indices = torch.argsort(attr_flat, descending=True)

    num_pixels = attr_flat.numel()
    fractions = np.linspace(0.0, 1.0, steps + 1)
    scores: list[float] = []

    with torch.no_grad():
        for frac in fractions:
            k = int(round(frac * num_pixels))
            x_mod = x.clone()
            if k > 0:
                idx = sorted_indices[:k]
                x_mod_flat = x_mod.view(1, 3, -1)
                base_flat = base.view(1, 3, -1)
                x_mod_flat[:, :, idx] = base_flat[:, :, idx]
            score_t, _ = detector(x_mod)
            scores.append(float(score_t.view(-1)[0].item()))

    audc = float(np.trapezoid(scores, fractions))
    return {
        "audc": audc,
        "fractions": [float(f) for f in fractions],
        "scores": scores,
    }


def consistency_add(
    detector: BaseDetector,
    normal_image: Tensor,
    anomalous_image: Tensor,
    attr_map: Tensor | np.ndarray,
    steps: int = 10,
) -> dict[str, Any]:
    """Addition test: progressively inserts top-attributed anomalous pixels into a normal image.
    Higher AUAC (Area Under Addition Curve) indicates a more faithful explanation.
    """
    device = (
        next(detector.parameters()).device
        if list(detector.parameters())
        else anomalous_image.device
    )
    x_norm = normal_image.to(device, dtype=torch.float32)
    if x_norm.ndim == 3:
        x_norm = x_norm.unsqueeze(0)

    x_anom = anomalous_image.to(device, dtype=torch.float32)
    if x_anom.ndim == 3:
        x_anom = x_anom.unsqueeze(0)

    if isinstance(attr_map, np.ndarray):
        attr_t = torch.from_numpy(attr_map).to(device, dtype=torch.float32)
    else:
        attr_t = attr_map.to(device, dtype=torch.float32)
    attr_flat = attr_t.reshape(-1)
    sorted_indices = torch.argsort(attr_flat, descending=True)

    num_pixels = attr_flat.numel()
    fractions = np.linspace(0.0, 1.0, steps + 1)
    scores: list[float] = []

    with torch.no_grad():
        for frac in fractions:
            k = int(round(frac * num_pixels))
            x_comp = x_norm.clone()
            if k > 0:
                idx = sorted_indices[:k]
                x_comp_flat = x_comp.view(1, 3, -1)
                x_anom_flat = x_anom.view(1, 3, -1)
                x_comp_flat[:, :, idx] = x_anom_flat[:, :, idx]
            score_t, _ = detector(x_comp)
            scores.append(float(score_t.view(-1)[0].item()))

    auac = float(np.trapezoid(scores, fractions))
    return {
        "auac": auac,
        "fractions": [float(f) for f in fractions],
        "scores": scores,
    }
```

#### Acceptance Test: `tests/test_explanation_metrics.py`
* Test with a synthetic $20 \times 20$ binary mask:
  * Perfect attribution (`attr == mask`) gives `correctness_auc == 1.0` and `correctness_sum == 1.0`.
  * Inverted attribution (`attr == 1 - mask`) gives `correctness_auc == 0.0` and `correctness_sum == 0.0`.
  * Run `consistency_del` and `consistency_add` with a dummy `BaseDetector` whose score is the mean pixel intensity inside the mask; verify `audc < auac` and both return valid dicts.
* **Acceptance Command:** `pytest tests/test_explanation_metrics.py -v` and update `weekly_progress/week5_abigx_and_xai_metrics.md`.

---

### STEP 20 (Week 6A): Explanation Benchmark & AFR Ablation Runner (`scripts/run_explanations.py`)

**Goal:** Build `scripts/run_explanations.py` and `configs/explanations.yaml` to evaluate all 5 explainers (`saliency`, `gradcam_vit`, `ig_black`, `ig_expected`, `abigx`) on anomalous test images, save comparison panels, and run the AFR hyperparameter ablation (`afr_steps` $\in \{5, 10, 20\}$, `afr_step_size` $\in \{0.01, 0.03, 0.05\}$).

* **CLI Flags:**
  `python scripts/run_explanations.py --config configs/explanations.yaml --category carpet --max_images 5 --device auto [--ablate_afr]`
* **Outputs Saved:**
  * `results/explanations/<category>/metrics_per_image.csv`
  * `results/explanations/summary_benchmark.csv` with columns:
    `category,explainer,correctness_auc,correctness_sum,consistency_del_audc,consistency_add_auac,afr_convergence_rate,sec_per_image`
  * `results/explanations/afr_ablation.csv` (when `--ablate_afr` is passed).
  * Multi-panel PNG visualizations under `results/explanations/<category>/panels/`.
* **Acceptance Command:** Run `python scripts/run_explanations.py --category carpet --max_images 3` and print the summary CSV table. Update `weekly_progress/week6_benchmark_and_vlm_text.md`.

---

### STEP 21 (Week 6B): Grounded VLM & OpenCLIP Language Explanation Module (`src/language/vlm_explainer.py`)

**Goal:** Generate structured, hallucination-checked natural-language explanations grounded in the detector's score and the explainer's attribution map.

#### Key Engineering Design (`src/language/vlm_explainer.py`)
To ensure your live demo works **100% offline on any Mac or Windows laptop without needing a 16 GB VRAM LLM server**, while also supporting an external VLM API (like Ollama `llava` / `qwen2.5vl` or OpenAI API) when available, implement a dual-mode `GroundedLanguageExplainer`:

1. **Spatial Region Extraction (`extract_defect_region`):**
   * Given `attr_map [240, 240]`, threshold at the `90th` percentile (`np.percentile(attr_map, 90)`).
   * Find the bounding box `(x_min, y_min, x_max, y_max)` of the largest connected component (using `scipy.ndimage.label`).
   * Compute center of mass $(c_x, c_y)$ in `[0, 240]`:
     * If $80 \le c_x \le 160$ and $80 \le c_y \le 160$: `"center"`
     * Else combine vertical (`"top"` if $c_y < 120$ else `"bottom"`) and horizontal (`"left"` if $c_x < 120$ else `"right"`) into `"top-left"`, `"top-right"`, `"bottom-left"`, or `"bottom-right"`.
   * Compute `area_percentage` = $100 \times \frac{\text{component pixels}}{240 \times 240}$.

2. **Hallucination Guardrail (Proposal Section 4.4 Mitigation):**
   * If `anomaly_score < anomaly_threshold` (default `0.45`):
     Immediately return `LanguageExplanation` with `is_anomalous=False`, `predicted_defect_type="none"`, `spatial_quadrant="none"`, `grounding_bbox=(0, 0, 0, 0)`, and `visual_evidence=f"The {category} sample appears normal (anomaly score {anomaly_score:.2f} is below the {anomaly_threshold:.2f} threshold). No concentrated surface or structural defect was detected."`

3. **Zero-Shot OpenCLIP Defect Classifier + Grounded Template (`backend="openclip_grounded"`, Default):**
   * Crops the input tensor `image[:, :, y_min:y_max, x_min:x_max]` (padded by 16 pixels for context), resizes to `(240, 240)`, and encodes it with the already-loaded `WinCLIPDetector` OpenCLIP image encoder.
   * Compares the crop embedding against candidate defect prompts:
     * `"cut"`: `"a photo of a surface cut or slash"`
     * `"hole"`: `"a photo of a puncture, hole, or missing material"`
     * `"scratch"`: `"a photo of a thin surface scratch or abrasion"`
     * `"contamination"`: `"a photo of foreign contamination, stain, or discoloration"`
     * `"crack"`: `"a photo of a structural crack or fracture"`
     * `"deformation"`: `"a photo of a bent, warped, or misplaced component"`
   * Picks the highest-probability defect class (`predicted_defect_type`, `defect_confidence`) and synthesizes a clear, operator-ready explanation citing the exact quadrant, bounding box coordinates, area share, and detector score.
   * Also supports `backend="ollama_vlm"` (optional HTTP POST to `http://localhost:11434/api/generate` with model `llava`) with automatic fallback to `"openclip_grounded"` if Ollama is not running.

* **Acceptance Command:** `pytest tests/test_language_explainer.py -v` and update `weekly_progress/week6_benchmark_and_vlm_text.md`.

---

### STEP 22 (Week 7A): Language Explanation Evaluation Suite (`src/metrics/language_eval.py` & `scripts/run_language_eval.py`)

**Goal:** Quantitatively evaluate the language explanations against MVTec AD ground truth:

1. **Defect-Type Semantic Accuracy (`compute_defect_type_accuracy`):**
   * Maps MVTec folder defect names (`cut`, `thread`, `hole`, `color`, `metal_contamination`, `scratch_neck`, `broken_large`, `bent_lead`, etc.) to canonical defect families (`cut`, `hole`, `scratch`, `contamination`, `crack`, `deformation`) and checks whether `predicted_defect_type` matches the ground-truth family (or matches via CLIP text embedding cosine similarity $\ge 0.72$).
2. **Region Grounding Agreement (`compute_region_grounding`):**
   * Computes ground-truth mask center-of-mass quadrant and bounding box.
   * Reports **Quadrant Match Accuracy** (boolean match between predicted quadrant and true mask quadrant) and **Bounding-Box IoU** (Intersection over Union between `grounding_bbox` and ground-truth mask bbox).
3. **Hallucination Rate (`compute_hallucination_rate`):**
   * On normal (`good`) images: False positive defect claim rate (`is_anomalous == True`).
   * On anomalous images: Unfounded spatial claim rate (when `grounding_bbox` has $0.0$ overlap with the ground-truth defect mask).
4. **Human Rating Sheet Exporter:**
   * Saves `results/language/human_rating_rubric.csv` pre-populated with image path, ground-truth defect, generated text explanation, and blank columns (`clarity_1_to_5`, `groundedness_1_to_5`, `actionability_1_to_5`) for Prabhjot and Sarhan to rate 20 samples.

* **Acceptance Command:** `pytest tests/test_language_metrics.py -v` and update `weekly_progress/week7_language_eval_and_frontend.md`.

---

### STEP 23 (Week 7B): Full-Stack Interactive Web Frontend (`app/streamlit_app.py`)

**Goal:** Build a polished, responsive, 4-tab Streamlit application (`app/streamlit_app.py`) that integrates **every module from Weeks 1 to 7** for next week's live demo.

#### Required Frontend Architecture (`app/streamlit_app.py`)
* **Sidebar Controls:**
  1. **Hardware Device Indicator:** Shows active device (`APPLE MPS`, `NVIDIA CUDA`, or `CPU`) + random seed lock.
  2. **Category Selector:** All 15 MVTec AD categories (with quick-select presets for `carpet`, `leather`, `tile`, `bottle`, `transistor`).
  3. **Detector Configuration:**
     * Model: `WinCLIP (ViT-B/16+)` | `AnomalyCLIP` | `PatchCore`
     * Mode: `Zero-Shot (k=0)` | `Few-Shot (k=1)` | `Few-Shot (k=2)` | `Few-Shot (k=4)`
     * Score Mode: `combined` | `combined_smooth` | `global` | `map_max`
  4. **Visual Explainer Selector:**
     * `ABIGX (AFR + Integrated Gradients)` (Default)
     * `Integrated Gradients (Expected Normal Baseline)`
     * `Integrated Gradients (Black Baseline)`
     * `ViT Grad-CAM (Attention Layer)`
     * `Saliency Map (Input Gradients)`
  5. **Image Input Mode:**
     * **Mode A:** Browse MVTec AD Test Dataset (Dropdown by `defect_type` + image index, automatically loads ground-truth mask for live metric grading).
     * **Mode B:** Upload Custom Image (`.png`, `.jpg`, `.jpeg`).

* **Main Interface — 4 Interactive Tabs:**
  * **Tab 1: Live Inspection & Grounded AI Report (Primary Demo View)**
    * **Top KPI Banner (4 Metrics):**
      1. `Anomaly Score` (e.g., `0.89` with color-coded `DEFECT DETECTED` / `PASS - NORMAL` badge)
      2. `Ground-Truth Verdict` (Matches MVTec label if dataset image selected)
      3. `Active Explainer` (`ABIGX` + Convergence badge)
      4. `Inference Latency` (in seconds)
    * **4-Column Visual Inspection Grid:**
      * Col 1: Original Input Image ($240 \times 240$) with Grounded Bounding Box overlay
      * Col 2: Detector Raw Anomaly Heatmap ($M$)
      * Col 3: XAI Attribution Heatmap Overlay ($A$)
      * Col 4: Ground-Truth Defect Mask (or High-Attribution Cropped Defect Zoom if custom upload)
    * **Grounded Natural-Language Operator Card:**
      * Displays the structured report from `GroundedLanguageExplainer`: Predicted Defect Type, Confidence, Spatial Quadrant, Area %, Visual Evidence paragraph, Recommended Action, and a **"Download Inspection JSON"** button.
  * **Tab 2: ABIGX Deep-Dive & Faithfulness Curves (Research Contribution View)**
    * Shows side-by-side: `(1) Original Defective Image` $\rightarrow$ `(2) AFR Reconstructed Normal Baseline` $\rightarrow$ `(3) Pixel Difference |x - x_AFR|` $\rightarrow$ `(4) Final ABIGX Attribution`.
    * Plots the **AFR PGD Optimization Trajectory** (Anomaly score dropping across PGD steps).
    * Interactive **"Run Live Faithfulness Check (DEL & ADD)"** button that computes and plots the Deletion Curve (`AUDC`) and Addition Curve (`AUAC`) + Correctness `AUC` and `SUM` on the current image in real time.
  * **Tab 3: Side-by-Side All-Explainers Arena**
    * One-click button to run `Saliency`, `ViT Grad-CAM`, `IG (Black)`, `IG (Expected)`, and `ABIGX` on the selected image and display all 5 heatmaps + their `Correctness AUC` and `Correctness SUM` scores in a single comparison grid.
  * **Tab 4: Project Benchmarks & Results Explorer**
    * Loads and displays interactive tables and bar charts from `results/` (Zero-Shot vs. Few-Shot detection tables, Explainer benchmark CSV, and Language evaluation metrics).

* **Smoke Test (`tests/test_frontend_smoke.py`):**
  * Verifies that `app/streamlit_app.py` compiles with `py_compile` and its helper functions execute cleanly on a dummy image tensor.
* **Acceptance Command:** `pytest tests/test_frontend_smoke.py -v` and update `weekly_progress/week7_language_eval_and_frontend.md`.

---

### STEP 24 (Week 8): Automated IEEE Report Artifacts & Next-Week Demo Playbook

**Goal:** Create `scripts/generate_report_artifacts.py` and `weekly_progress/week8_demo_playbook_and_ieee_report.md` so all experimental tables (Markdown + LaTeX) and presentation talking points are ready for the demo and final report.

* **Files to Create:**
  * `scripts/generate_report_artifacts.py`:
    * Reads all existing CSVs in `results/` (or generates a complete, clearly labeled baseline verification run on a sample subset if any CSV is missing) and outputs ready-to-paste IEEE LaTeX tables and Markdown tables into `results/report_tables/`:
      1. `table1_detection_baselines.md` / `.tex` (WinCLIP Zero-Shot, Few-Shot $k=1,2,4$, AnomalyCLIP, PatchCore).
      2. `table2_visual_explainers.md` / `.tex` (Saliency, Grad-CAM, IG-Black, IG-Expected, ABIGX across AUC, SUM, DEL, ADD).
      3. `table3_language_evaluation.md` / `.tex` (Defect Accuracy, Quadrant Grounding, BBox IoU, Hallucination Rate).
  * `weekly_progress/week8_demo_playbook_and_ieee_report.md`:
    * **5-Minute Live Demo Script** for Prabhjot and Sarhan (exact clicks in `streamlit run app/streamlit_app.py`, what to say on each tab, and how to explain ABIGX vs. standard IG to Dr. Shaon Shuvo and Dr. Prashanth Cheluvasai Ranga).
    * **10 Likely Supervisor Q&A Questions** with clear, beginner-friendly technical answers.
* **Acceptance Command:** Run `python scripts/generate_report_artifacts.py` and `pytest -v` across the entire repository.

---

## 4. COPY-PASTE PROMPTS FOR YOUR ANTIGRAVITY AGENT

Since Weeks 1–4 are already complete, paste one prompt at a time to your Antigravity agent (or tell it to run Steps 18–24 sequentially, stopping only if a test fails):

### Option A: "One-Go Autonomous Sprint" Prompt (Recommended to finish everything fast)
> Read `IMPLEMENTATION_PLAN_WEEKS_3_4.md` in full. Weeks 1–4 are already implemented. Execute **Steps 18 through 24** sequentially to complete the entire remaining project (Weeks 5–8 + Full-Stack Streamlit Frontend). After each step, run its `pytest` acceptance test, update the corresponding beginner guide in `weekly_progress/`, and automatically proceed to the next step as long as all tests pass. At the end of Step 24, run the full `pytest -v` suite and show me the complete test summary and the exact command to launch the Streamlit demo.

### Option B: Step-by-Step Prompts (If you want to review after each step)
* **Prompt 1 (Step 18):** *"Read `IMPLEMENTATION_PLAN_WEEKS_3_4.md` and implement **Step 18 (ABIGX Explainer: AFR Baseline Search + IG)** only: `src/explainers/abigx.py` and `tests/test_abigx.py`. Run `pytest tests/test_abigx.py -v`, update `weekly_progress/week5_abigx_and_xai_metrics.md`, and show me the output."*
* **Prompt 2 (Step 19):** *"Implement **Step 19 (Quantitative Explanation Metrics: Correctness AUC/SUM & Consistency DEL/ADD)** only: `src/metrics/explanation.py` and `tests/test_explanation_metrics.py`. Run `pytest tests/test_explanation_metrics.py -v`, update `weekly_progress/week5_abigx_and_xai_metrics.md`, and show me the output."*
* **Prompt 3 (Step 20):** *"Implement **Step 20 (Explanation Benchmark & AFR Ablation Runner)** only: `configs/explanations.yaml` and `scripts/run_explanations.py`. Run a quick verification test, update `weekly_progress/week6_benchmark_and_vlm_text.md`, and show me the output."*
* **Prompt 4 (Step 21):** *"Implement **Step 21 (Grounded Language Explanation Module)** only: `src/language/vlm_explainer.py`, `configs/language.yaml`, and `tests/test_language_explainer.py`. Run `pytest tests/test_language_explainer.py -v`, update `weekly_progress/week6_benchmark_and_vlm_text.md`, and show me the output."*
* **Prompt 5 (Step 22):** *"Implement **Step 22 (Language Explanation Evaluation Suite)** only: `src/metrics/language_eval.py`, `scripts/run_language_eval.py`, and `tests/test_language_metrics.py`. Run `pytest tests/test_language_metrics.py -v`, update `weekly_progress/week7_language_eval_and_frontend.md`, and show me the output."*
* **Prompt 6 (Step 23):** *"Implement **Step 23 (Full-Stack Interactive 4-Tab Streamlit Frontend)** only: `app/streamlit_app.py` and `tests/test_frontend_smoke.py`. Run `pytest tests/test_frontend_smoke.py -v`, update `weekly_progress/week7_language_eval_and_frontend.md`, and show me the output."*
* **Prompt 7 (Step 24):** *"Implement **Step 24 (Automated IEEE Report Artifacts & Demo Playbook)** only: `scripts/generate_report_artifacts.py` and `weekly_progress/week8_demo_playbook_and_ieee_report.md`. Run `python scripts/generate_report_artifacts.py` and the full `pytest -v` test suite, and show me the final results."*