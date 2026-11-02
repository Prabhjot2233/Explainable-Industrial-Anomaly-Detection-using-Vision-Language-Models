# COMP 8977 — Master Implementation Plan (Weeks 3–4)

**Project:** Explainable Industrial Anomaly Detection using Vision-Language Models (Team 12)  
**Target Agent:** Antigravity Autonomous Coding Agent  
**Scope:** Week 3 (Few-Shot WinCLIP, Prompt Ablation, AnomalyCLIP, PatchCore) & Week 4 (End-to-End Differentiability, Saliency, Integrated Gradients, ViT Grad-CAM, Visualization Tools)  
**Supported OS & Hardware:** Cross-Platform — macOS (Apple Silicon `mps` / `cpu`) & Windows (NVIDIA `cuda` / `cpu`)

---

## 0. CRITICAL DIRECTIVES FOR THE ANTIGRAVITY AGENT

1. **Pre-Flight Check (Verify Weeks 1–2 First):**
   * Before starting Step 11, run `pytest -v` on the existing Weeks 1–2 tests (`tests/test_utils_config.py`, `tests/test_metrics.py`, `tests/test_dataset.py`, `tests/test_winclip_shapes.py`, `tests/test_grad_flow.py`).
   * Do **NOT** break any existing interfaces in `src/data/mvtec.py`, `src/detectors/base.py`, or `src/metrics/detection.py`.
2. **Sequential Step-by-Step Execution (Steps 11 to 17):**
   * Build strictly in the order defined in **Section 4** (Step 11 $\rightarrow$ Step 17), **one step at a time**.
   * **STOP** after completing each step, run its exact `pytest` or CLI acceptance command, update the `weekly_progress/` beginner guide for that step, and show the terminal output to the user before moving on.
3. **Pure PyTorch Differentiable Forward Path (Zero NumPy/cv2 Inside Models or Explainers):**
   * Every detector (`WinCLIPDetector`, `AnomalyCLIPDetector`, `PatchCoreDetector`) MUST keep its `forward(x: Tensor) -> tuple[Tensor, Tensor]` method 100% in PyTorch.
   * **NEVER** place `with torch.no_grad():` inside `forward()` or `set_reference()`'s scoring path. Only place `with torch.no_grad():` inside runner scripts (`scripts/run_fewshot.py`, `scripts/run_baselines.py`) or inside `set_reference()` when pre-extracting static reference features from normal training images.
4. **Cross-Platform Mandate (macOS & Windows):**
   * Use `pathlib.Path` everywhere and `.as_posix()` when converting paths to strings.
   * Always specify `encoding="utf-8"` when opening YAML, CSV, JSON, or Markdown files.
   * Default `num_workers: 0` in all DataLoaders so Windows (`spawn`) and macOS (`mps`) never hang.
   * Use `get_device()` from `src/utils/device.py` (`cuda` $\rightarrow$ `mps` $\rightarrow$ `cpu`).
5. **Beginner-Friendly Weekly Tracking (`weekly_progress/`):**
   * You MUST create and update `weekly_progress/week3_detection_baselines.md` (for Steps 11–13) and `weekly_progress/week4_visual_explanations.md` (for Steps 14–17).
   * Write these guides so a complete beginner to Python and Machine Learning can understand every concept, math equation, and code file using simple real-world analogies.

---

## 1. DIRECTORY LAYOUT FOR WEEKS 3–4

Add these files to the existing Weeks 1–2 repository structure:

```text
.
├── configs/
│   ├── zeroshot.yaml                     # Existing (Weeks 1-2)
│   ├── fewshot.yaml                      # NEW (Step 11): Few-shot & prompt ablation config
│   ├── baselines.yaml                    # NEW (Step 12-13): PatchCore & AnomalyCLIP config
│   └── explanations_w4.yaml              # NEW (Step 14-17): Week 4 visual explainers config
├── weekly_progress/
│   ├── README.md                         # Update with Week 3 & Week 4 links
│   ├── glossary_for_beginners.md         # Update with Few-Shot, PatchCore, Saliency, IG, Grad-CAM
│   ├── week3_detection_baselines.md      # NEW: Plain-English guide for Steps 11, 12, 13
│   └── week4_visual_explanations.md      # NEW: Plain-English guide for Steps 14, 15, 16, 17
├── src/
│   ├── detectors/
│   │   ├── base.py                       # Existing BaseDetector contract
│   │   ├── prompts.py                    # UPDATE (Step 11): Add prompt_mode ablation support
│   │   ├── winclip.py                    # UPDATE (Step 11): Implement set_reference() & WinCLIP+ scoring
│   │   ├── patchcore.py                  # NEW (Step 12): Differentiable PyTorch PatchCore detector
│   │   └── anomalyclip.py                # NEW (Step 13): Object-agnostic prompt VLM detector
│   └── explainers/
│       ├── __init__.py                   # NEW (Step 14)
│       ├── base.py                       # NEW (Step 14): BaseExplainer contract & scalar wrapper
│       ├── saliency.py                   # NEW (Step 14): Input Gradient / Captum Saliency
│       ├── ig.py                         # NEW (Step 15): Integrated Gradients (Black & Expected-IG)
│       ├── gradcam_vit.py                # NEW (Step 16): ViT Grad-CAM & Attention Relevance
│       └── visualize.py                  # NEW (Step 17): Multi-panel heatmap overlay generator
├── scripts/
│   ├── run_fewshot.py                    # NEW (Step 11): Runs k={1,2,4} few-shot & prompt ablation
│   ├── run_baselines.py                  # NEW (Step 13): Runs PatchCore & AnomalyCLIP evaluation
│   └── run_visual_explanations.py        # NEW (Step 17): Runs Saliency, IG, Grad-CAM & saves panels
└── tests/
    ├── test_fewshot.py                   # NEW (Step 11)
    ├── test_patchcore.py                 # NEW (Step 12)
    ├── test_anomalyclip.py               # NEW (Step 13)
    ├── test_saliency_ig.py               # NEW (Step 14 & 15)
    └── test_gradcam_visualize.py         # NEW (Step 16 & 17)
```

---

## 2. FIXED INTERFACES & MATHEMATICAL SPECIFICATIONS

### 2.1 Prompt Ablation Modes (`src/detectors/prompts.py`)
Update `build_text_embedding(...)` to accept `prompt_mode: str = "full_ensemble"` with three valid modes:
1. `"single"`: Uses only 1 template (`"a photo of a {}."`) and 1 state per class (Normal: `"{}"`, Anomalous: `"damaged {}"`).
2. `"state_only"`: Uses 1 photo template (`"a photo of a {}."`) combined with all state words (all normal states vs. all anomalous states).
3. `"full_ensemble"`: Uses the full Cartesian product of all photo templates $\times$ all state words (the default WinCLIP Compositional Prompt Ensemble from Week 2).

*Cache key must include `prompt_mode`:* `results/cache/text_emb_<category>_<pretrained_tag>_<prompt_mode>.pt`.

---

### 2.2 WinCLIP+ Few-Shot Specification (`src/detectors/winclip.py`)
In Week 3, `WinCLIPDetector` is upgraded to **WinCLIP+** by implementing `set_reference(normal_images: Tensor | None)`:

1. **Reset Behavior:**
   * If `normal_images is None`, clear `self.ref_window_features = {}` and set `self.is_fewshot = False` (reverts to pure Zero-Shot WinCLIP).
2. **Storing Reference Features (`normal_images: Tensor` of shape `[K, 3, 240, 240]`):**
   * Under `with torch.no_grad():`, for each scale $s \in \text{self.window\_scales}$ (e.g., `[2, 3]`):
     * Extract sliding windows from `normal_images` using `F.unfold` (exact same window size and stride as `forward`).
     * Resize windows to `(240, 240)` in chunks of `window_chunk_size`, encode via `self.model.encode_image(...)`, and L2-normalize along feature dimension $D$.
     * Store the tensor in `self.ref_window_features[s]` with shape `[K * N_win_s, D]`, detached on `self.device`.
   * Set `self.is_fewshot = True`.
3. **Few-Shot Forward Pass (`forward(x: Tensor)` with Gradients Enabled):**
   * Compute Zero-Shot global score $s_{\text{global}} \in \mathbb{R}^{B}$ and Zero-Shot multi-scale text anomaly map $M_{\text{text}} \in \mathbb{R}^{B \times 240 \times 240}$ exactly as in Week 2.
   * If `self.is_fewshot` is `True`:
     * For each scale $s \in \text{self.window\_scales}$, take the L2-normalized test window embeddings $\hat{\mathbf{F}}_s \in \mathbb{R}^{B \times N_{\text{win}, s} \times D}$ (already computed during Stage B window scoring — **do NOT encode the windows twice!**).
     * Compute cosine similarity against `self.ref_window_features[s]` ($\mathbf{R}_s \in \mathbb{R}^{M_s \times D}$):
       $$\text{Sim}_s = \hat{\mathbf{F}}_s @ \mathbf{R}_s^\top \in \mathbb{R}^{B \times N_{\text{win}, s} \times M_s}$$
     * Convert similarity to visual anomaly score per window:
       * If `score_mode == "combined_smooth"` (used for Week 4 gradients), use smooth maximum over reference patches with temperature $\tau_{\text{ref}} = 50.0$:
         $$S_{\text{vis}, s} = \frac{1}{2}\left(1.0 - \frac{1}{\tau_{\text{ref}}}\text{logsumexp}(\tau_{\text{ref}} \cdot \text{Sim}_s, \text{dim}=-1)\right)$$
       * Otherwise (standard Week 3 evaluation), use exact max:
         $$S_{\text{vis}, s} = \frac{1}{2}\left(1.0 - \max_{m} \text{Sim}_s(\cdot, m)\right)$$
     * Aggregate $S_{\text{vis}, s}$ onto `[B, 240, 240]` using the exact same clamped harmonic mean (`F.fold`) as the text map, and average across scales $s$ to get $M_{\text{vis}} \in \mathbb{R}^{B \times 240 \times 240}$.
     * Fuse zero-shot text map and few-shot visual map with weight $\alpha_{\text{vis}} = 0.5$:
       $$M_{\text{final}} = (1 - \alpha_{\text{vis}}) M_{\text{text}} + \alpha_{\text{vis}} M_{\text{vis}}$$
   * Compute final image score from $s_{\text{global}}$ and $M_{\text{final}}$ according to `self.score_mode` (`"global"`, `"map_max"`, `"combined"`, or `"combined_smooth"`).

---

### 2.3 PatchCore Reference Detector (`src/detectors/patchcore.py`)
`PatchCoreDetector(BaseDetector)` provides the non-VLM memory-bank baseline required by Proposal Section 4.2:

1. **Backbone:** Load `torchvision.models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)` in `float32`, `.eval()`, with all parameters frozen (`requires_grad_(False)`).
2. **Input Re-Normalization (Critical):**
   * `MVTecDataset` outputs images normalized with CLIP mean/std. Inside `PatchCoreDetector.forward`, first denormalize from CLIP mean/std back to `[0, 1]` and re-normalize with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)` using pure PyTorch tensor operations so gradients remain intact!
3. **Feature Extraction:**
   * Extract intermediate feature maps from `layer2` (shape `[B, 512, 30, 30]`) and `layer3` (shape `[B, 1024, 15, 15]`).
   * Apply $3 \times 3$ average pooling (`F.avg_pool2d(..., kernel_size=3, stride=1, padding=1)`) to capture local neighborhood context.
   * Bilinearly upsample `layer3` to `(30, 30)` and concatenate along the channel dimension $\rightarrow$ `[B, 1536, 30, 30]`. L2-normalize across the 1536 channel dimension.
4. **Memory Bank (`set_reference(normal_images)`):**
   * Extract `[K * 900, 1536]` patch vectors from normal training images.
   * Subsample with deterministic random/greedy coreset ratio `coreset_ratio: float = 0.1` (default max `2000` patches so Mac/Windows runs in seconds) using a fixed-seed generator. Store `self.memory_bank` of shape `[M, 1536]`.
5. **Forward Scoring (`forward(x)`):**
   * Compute squared Euclidean distance (or cosine distance $1 - \hat{\mathbf{p}} @ \mathbf{M}^\top$) between each of the $30 \times 30 = 900$ test patch vectors and `self.memory_bank`.
   * Find minimum distance per patch $\rightarrow$ `[B, 1, 30, 30]`, bilinearly upsample to `[B, 240, 240]`, and apply differentiable 2D Gaussian smoothing ($\sigma = 4.0$).
   * Image score is `anomaly_map.amax(dim=(1, 2))` (or smooth LogSumExp when `score_mode == "combined_smooth"`).

---

### 2.4 AnomalyCLIP Reference Detector (`src/detectors/anomalyclip.py`)
`AnomalyCLIPDetector(BaseDetector)` implements object-agnostic prompt anomaly detection over the OpenCLIP `ViT-B-16-plus-240` backbone:
1. Instead of category-specific engineering, it builds **object-agnostic canonical prompts** (`"a photo of a flawless object."`, `"a photo of a normal object without defect."` vs. `"a photo of a damaged object with a defect."`, `"a photo of a broken anomalous object."`) combined with optional learnable/initialized prompt offset vectors $\Delta \mathbf{T} \in \mathbb{R}^{2 \times D}$.
2. Extracts dense patch-level token features directly from the ViT visual backbone (or dense sliding windows with `window_scales=[1, 2]`), projects them to the text embedding space, computes temperature-scaled softmax probability of the anomalous class at each spatial location, and interpolates to `[B, 240, 240]`.
3. Adheres strictly to `BaseDetector.forward(x) -> (score [B], anomaly_map [B, 240, 240])` with full input gradient support.

---

### 2.5 Visual Explainer Contract (`src/explainers/base.py`)
All Week 4 explainers inherit from `BaseExplainer`:

```python
from abc import ABC, abstractmethod
import torch
from torch import Tensor, nn
from src.detectors.base import BaseDetector

class DifferentiableScoreWrapper(nn.Module):
    """Wraps a BaseDetector so forward(x) returns ONLY a 1D tensor of scalar scores [B].
    Automatically switches detector score_mode to 'combined_smooth' during attribution
    and restores the original score_mode afterward.
    """
    def __init__(self, detector: BaseDetector, target_score_mode: str = "combined_smooth"):
        super().__init__()
        self.detector = detector
        self.target_score_mode = target_score_mode

    def forward(self, x: Tensor) -> Tensor:
        old_mode = getattr(self.detector, "score_mode", None)
        try:
            if old_mode is not None:
                self.detector.score_mode = self.target_score_mode
            scores, _ = self.detector(x)
            return scores
        finally:
            if old_mode is not None:
                self.detector.score_mode = old_mode

class BaseExplainer(ABC):
    def __init__(self, detector: BaseDetector, device: torch.device):
        self.detector = detector
        self.device = device
        self.score_wrapper = DifferentiableScoreWrapper(detector)

    @staticmethod
    def aggregate_and_normalize(raw_attr: Tensor, eps: float = 1e-8) -> Tensor:
        """Converts raw attribution [B, 3, H, W] -> [B, H, W] in [0.0, 1.0].
        Takes L2 norm across color channels (dim=1), then min-max normalizes per image.
        """
        spatial = torch.linalg.vector_norm(raw_attr, ord=2, dim=1)  # [B, H, W]
        b = spatial.shape[0]
        flat = spatial.view(b, -1)
        mins = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1)
        maxs = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1)
        return (spatial - mins) / (maxs - mins + eps)

    @abstractmethod
    def explain(
        self,
        image: Tensor,                        # [1, 3, 240, 240]
        reference_normals: Tensor | None = None,  # Optional [K, 3, 240, 240]
    ) -> dict[str, Tensor | dict]:
        """Returns:
        {
            "attribution_map": Tensor [1, 240, 240] float32 in [0, 1],
            "raw_attribution": Tensor [1, 3, 240, 240] float32,
            "baseline": Tensor [1, 3, 240, 240] float32 (or None),
            "meta": dict
        }
        """
        raise NotImplementedError
```

---

## 3. REQUIREMENTS FOR `weekly_progress/` GUIDES (WEEKS 3 & 4)

As you complete Steps 11–17, create and update:
1. `weekly_progress/week3_detection_baselines.md`
2. `weekly_progress/week4_visual_explanations.md`
3. Update `weekly_progress/glossary_for_beginners.md` and `weekly_progress/README.md`.

Every step entry in those Markdown files MUST contain:
* **1. Plain-English Analogy:** (e.g., explain Few-Shot as giving an inspector 2 photos of a factory's brand-new carpet design before starting their shift; explain Integrated Gradients as slowly turning up a dimmer switch from a dark room to full brightness and watching when the inspector notices the scratch).
* **2. How the Math Works (Step-by-Step):** Explain every formula without jargon.
* **3. Files Created / Updated & What Each Function Does.**
* **4. How to Run & Expected Terminal Output** (for both macOS and Windows).

---

## 4. STEP-BY-STEP EXECUTION PLAN & ACCEPTANCE TESTS (STEPS 11–17)

---

### STEP 11 (Week 3 — Part 1): Few-Shot WinCLIP (`k = 1, 2, 4`) & Prompt Ablation
* **Owner in Proposal:** Prabhjot Singh
* **Files to Create / Update:**
  1. Update `src/detectors/prompts.py`: Add `prompt_mode: str = "full_ensemble"` (`"single"`, `"state_only"`, `"full_ensemble"`).
  2. Update `src/detectors/winclip.py`: Implement `set_reference(normal_images: Tensor | None)` and fuse $M_{\text{vis}}$ with $M_{\text{text}}$ as specified in Section 2.2.
  3. Create `configs/fewshot.yaml`: Defines `k_shots_list: [0, 1, 2, 4]`, `prompt_modes: ["single", "state_only", "full_ensemble"]`, `categories: ["carpet", "leather", "tile", "bottle", "transistor"]`.
  4. Create `scripts/run_fewshot.py`:
     * CLI flags: `--config configs/fewshot.yaml`, `--category <name|all>`, `--k_shots <int|all>`, `--prompt_mode <mode|all>`, `--max_images <int|None>`, `--device {auto,mps,cpu,cuda}`, `--overwrite`.
     * Deterministically loads $k$ normal images from `MVTecDataset(root, category, split="train")` using `seed`.
     * Calls `detector.set_reference(ref_images)`.
     * Evaluates on `split="test"` under `with torch.no_grad():` and writes results to `results/fewshot/fewshot_metrics.csv` and `results/fewshot/prompt_ablation.csv`.
  5. Create `tests/test_fewshot.py`:
     * Tests all 3 `prompt_mode` options produce `[2, D]` unit-norm embeddings and that `"single"` differs from `"full_ensemble"`.
     * Tests `detector.set_reference(ref_imgs)` with `k=2` changes the output anomaly map compared to zero-shot (`set_reference(None)`).
     * Tests input gradient flow (`scores.sum().backward()`) works when `is_fewshot=True`.
* **Acceptance Command:**
  ```bash
  pytest tests/test_fewshot.py -v
  ```
* **Documentation Update:** Create `weekly_progress/week3_detection_baselines.md` (Step 11 section).

---

### STEP 12 (Week 3 — Part 2): Differentiable PatchCore Reference Baseline
* **Owner in Proposal:** Sarhan Kapadiya
* **Files to Create / Update:**
  1. Create `src/detectors/patchcore.py`: Implement `PatchCoreDetector(BaseDetector)` as specified in Section 2.3.
  2. Create `tests/test_patchcore.py`:
     * Instantiates `PatchCoreDetector` on target device.
     * Calls `set_reference(normal_images)` with `2` synthetic `[3, 240, 240]` images and verifies `memory_bank` is populated.
     * Runs `forward(x)` on a batch of 2 images with `requires_grad=True`, checks output shapes `(score [2], map [2, 240, 240])`, calls `score.sum().backward()`, and asserts `x.grad` is finite and non-zero.
     * Creates a synthetic defective image (injecting a high-contrast square patch into a normal image) and asserts the defective image receives a strictly higher anomaly score than the normal image.
* **Acceptance Command:**
  ```bash
  pytest tests/test_patchcore.py -v
  ```
* **Documentation Update:** Update `weekly_progress/week3_detection_baselines.md` (Step 12 section).

---

### STEP 13 (Week 3 — Part 3): AnomalyCLIP Detector & Baselines Runner
* **Owner in Proposal:** Sarhan Kapadiya
* **Files to Create / Update:**
  1. Create `src/detectors/anomalyclip.py`: Implement `AnomalyCLIPDetector(BaseDetector)` as specified in Section 2.4.
  2. Create `configs/baselines.yaml`: Configuration for running `winclip_zeroshot`, `winclip_4shot`, `anomalyclip`, and `patchcore`.
  3. Create `scripts/run_baselines.py`:
     * CLI flags: `--config configs/baselines.yaml`, `--detector {all,anomalyclip,patchcore,winclip}`, `--category <name|all>`, `--max_images <int|None>`, `--device {auto,mps,cpu,cuda}`.
     * Writes unified comparison table to `results/baselines/detector_comparison.csv` with columns: `detector,category,k_shots,image_auroc,pixel_auroc,pro,sec_per_image,device`.
  4. Create `tests/test_anomalyclip.py`:
     * Tests `AnomalyCLIPDetector` output shapes `([B], [B, 240, 240])` and finite input gradients.
* **Acceptance Command:**
  ```bash
  pytest tests/test_anomalyclip.py -v
  ```
* **Documentation Update:** Complete `weekly_progress/week3_detection_baselines.md` (Step 13 section).

---

### STEP 14 (Week 4 — Part 1): Base Explainer, Differentiable Score Wrapper & Saliency
* **Owner in Proposal:** Prabhjot Singh
* **Files to Create / Update:**
  1. Verify `captum` is installed (`pip install captum` if missing).
  2. Create `src/explainers/__init__.py` and `src/explainers/base.py`: Implement `DifferentiableScoreWrapper` and `BaseExplainer` as specified in Section 2.5.
  3. Create `src/explainers/saliency.py`:
     * Implement `SaliencyExplainer(BaseExplainer)` using `captum.attr.Saliency(self.score_wrapper)`.
     * Ensure `image = image.detach().clone().to(self.device).requires_grad_(True)` inside `explain()`.
     * Compute `raw_attr = saliency.attribute(image, abs=False)` and normalize via `self.aggregate_and_normalize(raw_attr)`.
  4. Create `tests/test_saliency_ig.py` (Part A — Saliency tests):
     * Tests `DifferentiableScoreWrapper` returns `[B]` 1D tensor and restores `detector.score_mode`.
     * Tests `SaliencyExplainer.explain()` returns `attribution_map` of shape `[1, 240, 240]` with values in `[0.0, 1.0]`, finite values, and non-zero attribution mass.
* **Acceptance Command:**
  ```bash
  pytest tests/test_saliency_ig.py -v -k "saliency or wrapper"
  ```
* **Documentation Update:** Create `weekly_progress/week4_visual_explanations.md` (Step 14 section).

---

### STEP 15 (Week 4 — Part 2): Integrated Gradients (Black Baseline & Expected-IG Normal Baselines)
* **Owner in Proposal:** Prabhjot Singh
* **Files to Create / Update:**
  1. Create `src/explainers/ig.py`:
     * Implement `IntegratedGradientsExplainer(BaseExplainer)` using `captum.attr.IntegratedGradients(self.score_wrapper)`.
     * Constructor parameters: `baseline_mode: str = "black"` (`"black"` or `"expected"`), `n_steps: int = 20`, `internal_batch_size: int = 4` (keeps memory low on Mac MPS and Windows GPUs).
     * When `baseline_mode == "black"`:
       * Baseline is `torch.zeros_like(image)` (which represents the mean CLIP color in normalized space; also support `"true_black"` which is CLIP-normalized `[0,0,0]`).
       * Compute `raw_attr = ig.attribute(image, baselines=baseline, n_steps=self.n_steps, internal_batch_size=self.internal_batch_size)`.
       * Compute completeness delta `delta = abs(raw_attr.sum().item() - (score(image) - score(baseline)).item())` and store in `meta["completeness_delta"]`.
     * When `baseline_mode == "expected"` (Expected Integrated Gradients):
       * Requires `reference_normals: Tensor` of shape `[K, 3, 240, 240]` (if `None`, raise a clear `ValueError`).
       * Loops over each of the $K$ normal baseline images $b_k$, runs `ig.attribute(image, baselines=b_k, n_steps=max(10, self.n_steps // K), internal_batch_size=self.internal_batch_size)`, averages the raw attributions across the $K$ baselines, and normalizes to `[0, 1]`.
  2. Update `tests/test_saliency_ig.py` (Part B — IG tests):
     * Tests `IntegratedGradientsExplainer` in `"black"` mode with `n_steps=8`, verifying output shapes, range `[0, 1]`, and that `meta["completeness_delta"] < 0.05` (Completeness Axiom check).
     * Tests `IntegratedGradientsExplainer` in `"expected"` mode with `K=2` reference normal images (`n_steps=6`), verifying output shapes and that `"expected"` attribution differs from `"black"` attribution.
* **Acceptance Command:**
  ```bash
  pytest tests/test_saliency_ig.py -v
  ```
* **Documentation Update:** Update `weekly_progress/week4_visual_explanations.md` (Step 15 section).

---

### STEP 16 (Week 4 — Part 3): ViT Grad-CAM / Attention-Based Relevance (`src/explainers/gradcam_vit.py`)
* **Owner in Proposal:** Sarhan Kapadiya
* **Files to Create / Update:**
  1. Create `src/explainers/gradcam_vit.py`:
     * Implement `ViTGradCAMExplainer(BaseExplainer)` tailored for `open_clip` Vision Transformer (`ViT-B-16-plus-240`, where $240 / 16 = 15 \times 15 = 225$ spatial patch tokens + 1 class token = 226 tokens).
     * Register a forward hook and a full backward hook (`register_full_backward_hook`) on `detector.model.visual.transformer.resblocks[-1].ln_1` (or `resblocks[-1]`).
     * Run a single forward + backward pass on the global/smooth anomaly score.
     * Extract spatial patch activations $\mathbf{A} \in \mathbb{R}^{1 \times 225 \times D_{\text{vit}}}$ and gradients $\mathbf{G} \in \mathbb{R}^{1 \times 225 \times D_{\text{vit}}}$ (slicing off index `0`, the `[CLS]` token).
     * Compute channel weights $\mathbf{w} = \text{mean}(\mathbf{G}, \text{dim}=1, \text{keepdim}=True)$ and weighted patch relevance:
       $$\text{CAM}_{\text{patch}} = \text{ReLU}\left(\sum_{d} \mathbf{w}_d \cdot \mathbf{A}_d\right) \in \mathbb{R}^{1 \times 1 \times 15 \times 15}$$
     * Upsample $\text{CAM}_{\text{patch}}$ to `(240, 240)` via `F.interpolate(..., mode="bicubic", align_corners=False)`, clamp to $\ge 0$, and min-max normalize to `[0.0, 1.0]`.
     * Always remove hooks in a `finally:` block so the detector is never left with dangling hooks!
  2. Create `tests/test_gradcam_visualize.py` (Part A — ViT Grad-CAM test):
     * Tests `ViTGradCAMExplainer.explain()` returns `[1, 240, 240]` attribution map in `[0, 1]` and verifies no hooks remain attached to `resblocks[-1]` after `explain()` returns.
* **Acceptance Command:**
  ```bash
  pytest tests/test_gradcam_visualize.py -v -k "gradcam"
  ```
* **Documentation Update:** Update `weekly_progress/week4_visual_explanations.md` (Step 16 section).

---

### STEP 17 (Week 4 — Part 4): Heatmap Visualization Tool & Week 4 Runner
* **Owner in Proposal:** Sarhan Kapadiya
* **Files to Create / Update:**
  1. Create `src/explainers/visualize.py`:
     * `denormalize_clip_image(img_tensor: Tensor) -> np.ndarray`: Converts `[3, H, W]` CLIP-normalized tensor back to `[H, W, 3]` float32 NumPy RGB array in `[0.0, 1.0]`.
     * `create_heatmap_overlay(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> np.ndarray`: Blends a `[H, W]` `[0, 1]` map onto the RGB image using `matplotlib.cm`.
     * `save_explanation_comparison_grid(image: Tensor, gt_mask: Tensor, anomaly_map: Tensor, attributions: dict[str, Tensor], save_path: str | Path, title: str = "") -> Path`:
       * Renders and saves a publication-quality figure with columns:
         `Original RGB` | `Ground Truth Mask` | `Detector Anomaly Map` | `Saliency` | `ViT Grad-CAM` | `IG (Black)` | `Expected-IG (Normal)`
       * Uses `matplotlib` (`Agg` backend so it never pops up blocking GUI windows on Mac or Windows) and saves at `150 DPI`.
  2. Create `configs/explanations_w4.yaml` and `scripts/run_visual_explanations.py`:
     * CLI flags: `--config configs/explanations_w4.yaml`, `--category <name>`, `--max_images <int>` (default `5`), `--device {auto,mps,cpu,cuda}`.
     * Loads defective test images (`label == 1`) from `MVTecDataset`, runs `WinCLIPDetector`, runs all 4 Week 4 explainers (`Saliency`, `ViT_GradCAM`, `IG_Black`, `IG_Expected`), saves the `.npz` attribution maps and `.png` visual comparison grids under `results/explanations_w4/<category>/`, and records runtime per explainer in `results/explanations_w4/runtime_summary.csv`.
  3. Update `tests/test_gradcam_visualize.py` (Part B — Visualization test):
     * Calls `save_explanation_comparison_grid` using synthetic tensors into a `tmp_path` file and asserts the `.png` file exists and has size `> 10 KB`.
* **Acceptance Command:**
  ```bash
  pytest tests/ -v
  ```
* **Documentation Update:** Finalize `weekly_progress/week4_visual_explanations.md` and update `weekly_progress/README.md` and `weekly_progress/glossary_for_beginners.md`.

---

## 5. COPY-PASTE PROMPTS FOR YOUR ANTIGRAVITY AGENT (ONE PER STEP)

Once Weeks 1–2 are complete, feed these prompts **one at a time** to your Antigravity agent:

* **Prompt for Step 11:**
  > "Read `IMPLEMENTATION_PLAN_WEEKS_3_4.md`. Execute **Step 11 (Few-Shot WinCLIP & Prompt Ablation)** only. Update `src/detectors/prompts.py` and `src/detectors/winclip.py`, and create `configs/fewshot.yaml`, `scripts/run_fewshot.py`, and `tests/test_fewshot.py`. Run `pytest tests/test_fewshot.py -v`, create `weekly_progress/week3_detection_baselines.md` (Step 11 section) written for a complete beginner, and show me the test output."

* **Prompt for Step 12:**
  > "Read `IMPLEMENTATION_PLAN_WEEKS_3_4.md`. Execute **Step 12 (Differentiable PatchCore Baseline)** only. Create `src/detectors/patchcore.py` and `tests/test_patchcore.py`. Run `pytest tests/test_patchcore.py -v`, update `weekly_progress/week3_detection_baselines.md` (Step 12 section), and show me the test output."

* **Prompt for Step 13:**
  > "Read `IMPLEMENTATION_PLAN_WEEKS_3_4.md`. Execute **Step 13 (AnomalyCLIP Detector & Baselines Runner)** only. Create `src/detectors/anomalyclip.py`, `configs/baselines.yaml`, `scripts/run_baselines.py`, and `tests/test_anomalyclip.py`. Run `pytest tests/test_anomalyclip.py -v`, update `weekly_progress/week3_detection_baselines.md` (Step 13 section), and show me the test output."

* **Prompt for Step 14:**
  > "Read `IMPLEMENTATION_PLAN_WEEKS_3_4.md`. Execute **Step 14 (Base Explainer, Differentiable Score Wrapper & Saliency)** only. Create `src/explainers/__init__.py`, `src/explainers/base.py`, `src/explainers/saliency.py`, and `tests/test_saliency_ig.py` (Saliency & Wrapper tests). Run `pytest tests/test_saliency_ig.py -v`, create `weekly_progress/week4_visual_explanations.md` (Step 14 section), and show me the test output."

* **Prompt for Step 15:**
  > "Read `IMPLEMENTATION_PLAN_WEEKS_3_4.md`. Execute **Step 15 (Integrated Gradients — Black & Expected-IG Baselines)** only. Create `src/explainers/ig.py` and add the IG tests to `tests/test_saliency_ig.py`. Run `pytest tests/test_saliency_ig.py -v`, update `weekly_progress/week4_visual_explanations.md` (Step 15 section), and show me the test output."

* **Prompt for Step 16:**
  > "Read `IMPLEMENTATION_PLAN_WEEKS_3_4.md`. Execute **Step 16 (ViT Grad-CAM / Attention Relevance)** only. Create `src/explainers/gradcam_vit.py` and `tests/test_gradcam_visualize.py` (Grad-CAM tests). Run `pytest tests/test_gradcam_visualize.py -v`, update `weekly_progress/week4_visual_explanations.md` (Step 16 section), and show me the test output."

* **Prompt for Step 17:**
  > "Read `IMPLEMENTATION_PLAN_WEEKS_3_4.md`. Execute **Step 17 (Heatmap Visualization Tool & Week 4 Runner)** only. Create `src/explainers/visualize.py`, `configs/explanations_w4.yaml`, `scripts/run_visual_explanations.py`, and complete `tests/test_gradcam_visualize.py`. Run the full test suite `pytest -v`, finalize `weekly_progress/week4_visual_explanations.md` and `weekly_progress/glossary_for_beginners.md`, and show me the full pytest output."