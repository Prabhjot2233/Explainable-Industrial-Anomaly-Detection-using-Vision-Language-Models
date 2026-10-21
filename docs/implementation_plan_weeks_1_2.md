# COMP 8977 — Master Implementation Plan (Weeks 1–2)
**Project:** Explainable Industrial Anomaly Detection using Vision-Language Models (Team 12)  
**Target Agent:** Antigravity Autonomous Coding Agent  
**Supported OS & Hardware:** Cross-Platform — macOS (Apple Silicon MPS / CPU) & Windows (NVIDIA CUDA / CPU)

---

## 0. CRITICAL DIRECTIVES FOR THE ANTIGRAVITY AGENT

1. **Sequential Execution & Gatekeeping:**
   - You MUST build this project in the exact 10-step order defined in **Section 5**.
   - **STOP** after completing each step, run its acceptance test via terminal, verify the output, update the `weekly_progress/` beginner guide for that step, and show the test output to the user before moving to the next step.
2. **Beginner-Friendly Weekly Tracking Folder (`weekly_progress/`):**
   - You MUST create and maintain a dedicated folder named `weekly_progress/` at the root of the repository.
   - Inside this folder, maintain beginner-friendly documentation that explains **what** was built, **why** it was needed, **how** the math and code work in plain English (using real-world analogies), and **how to run/test it**. Even a complete beginner to Python, PyTorch, or Computer Vision must be able to read these files and understand the project end-to-end.
3. **Pure PyTorch Forward Path (Week 4 Differentiability Requirement):**
   - Write the detector forward pass (`WinCLIPDetector.forward`) strictly in PyTorch.
   - **NO NumPy or OpenCV (`cv2`)** operations are allowed inside `forward()`.
   - **NO `torch.no_grad()`** inside `BaseDetector` or `WinCLIPDetector`. Place `with torch.no_grad():` strictly inside the runner scripts (`scripts/run_zeroshot.py`, `scripts/compare_checkpoints.py`).
   - *Reason:* In Week 4, we compute Integrated Gradients (IG) and Adversarial Fault Reconstruction (AFR / ABIGX), which require backpropagation of gradients from the anomaly score all the way back to the input image pixels.
4. **Cross-Platform Compatibility (macOS + Windows):**
   - Always use `pathlib.Path` for file system operations. When storing file paths in dictionaries, CSVs, or caches, normalize them with `.as_posix()` so `/` vs `\` never causes bugs across macOS and Windows.
   - Always specify `encoding="utf-8"` when opening text, YAML, CSV, or Markdown files to avoid Windows `cp1252` decoding crashes.
   - Default `num_workers: 0` in PyTorch `DataLoader` configs so multiprocessing does not hang on Windows (`spawn`) or leak shared memory on macOS (`mps`).
5. **No Hallucinated Hyperparameters:**
   - Where this spec says `VERIFY`, mark the code with `# VERIFY against WinCLIP paper` and log the item in `docs/open_questions.md`.
6. **Strict Interface Contracts:**
   - Never silently alter the function signatures, tensor shapes, or dictionary keys specified in **Section 3**.

---

## 1. SYSTEM ARCHITECTURE & DESIGN DECISIONS

| Parameter / Component | Standard / Default | Engineering Rationale |
| :--- | :--- | :--- |
| **Backbone** | `open_clip` `ViT-B-16-plus-240` | Pretrained `laion400m_e32` (compare against `laion400m_e31` in Step 10). |
| **Resolution** | $240 \times 240$ | Applied identically to input images, ground-truth masks, and output anomaly maps. |
| **Start Categories** | `carpet`, `leather`, `tile`, `bottle`, `transistor` | Covers 3 textures and 2 objects before scaling to all 15 MVTec AD categories. |
| **Device Selection** | `auto` (`cuda` $\rightarrow$ `mps` $\rightarrow$ `cpu`) | Supports Windows NVIDIA GPUs (`cuda`), Mac Apple Silicon (`mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1`), and CPU fallback. |
| **Precision** | `torch.float32` | Required for stable gradient accumulation in Week 4 and consistent MPS/CPU operations. |
| **Score Modes** | `global`, `map_max`, `combined`, `combined_smooth` | `combined` uses exact max for Week 2 eval; `combined_smooth` uses LogSumExp ($\tau=20.0$) so spatial gradients flow smoothly in Week 4. |
| **Harmonic Mean Stability** | Clamp window scores to $\ge 10^{-6}$ | Prevents division-by-zero or `NaN` gradients when normal regions have near-zero anomaly scores. |
| **Seeds** | `0` everywhere | Seed Python `random`, `numpy`, `torch` (CPU, CUDA, MPS), and DataLoader generators. |

---

## 2. REPOSITORY DIRECTORY LAYOUT

Add the following structure to the existing repository (do not add extra CI workflows or unrelated scaffolding):

```text
.
├── configs/
│   └── zeroshot.yaml
├── docs/
│   ├── open_questions.md
│   └── lit/                        # Optional Week 1 literature summaries
├── weekly_progress/                # MANDATORY: Beginner-friendly tracking & guides
│   ├── README.md                   # Overview of the project & how to read the weekly guides
│   ├── week1_foundation_and_env.md # Explains Steps 1, 2, 3 in plain English
│   ├── week2_data_and_winclip.md   # Explains Steps 4, 5, 6, 7, 8, 9, 10 in plain English
│   └── glossary_for_beginners.md   # Explains CLIP, ViT, AUROC, PRO, Gradients, etc.
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── mvtec.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── prompts.py
│   │   └── winclip.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── detection.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── device.py
│       └── seed.py
├── scripts/
│   ├── check_env.py
│   ├── run_zeroshot.py
│   └── compare_checkpoints.py
├── tests/
│   ├── test_utils_config.py
│   ├── test_metrics.py
│   ├── test_dataset.py
│   ├── test_winclip_shapes.py
│   ├── test_grad_flow.py
│   └── test_device_consistency.py
└── results/                        # Git-ignored output directory
    └── cache/                      # Cached text embeddings & resized masks
```

---

## 3. FIXED INTERFACES (CONTRACTS)

### 3.1 Dataset Item (`src/data/mvtec.py`)
`MVTecDataset(root, category, split="test", image_size=240)`
- `split`: `"train"` (normal only) or `"test"` (normal + defective).
- `__getitem__(idx)` MUST return a `dict`:
  - `"image"`: `torch.Tensor` of shape `[3, 240, 240]`, dtype `torch.float32`, normalized with CLIP mean `(0.48145466, 0.4578275, 0.40821073)` and std `(0.26862954, 0.26130258, 0.27577711)` applied **exactly once**.
  - `"mask"`: `torch.Tensor` of shape `[1, 240, 240]`, dtype `torch.float32`, strictly binary in `{0.0, 1.0}` (resized with nearest-neighbor interpolation and binarized at `> 0.5`). All zeros for normal (`"good"`) images.
  - `"label"`: `int` (`0` for normal, `1` for anomalous).
  - `"category"`: `str` (e.g., `"carpet"`).
  - `"defect_type"`: `str` (`"good"` for normal, or folder name like `"cut"`, `"hole"`, etc.).
  - `"path"`: `str` (normalized POSIX path string via `Path(...).as_posix()`).
- Provide `get_loader(root, category, split, image_size=240, batch_size=1, num_workers=0, seed=0)` with a fixed-seed `torch.Generator` and `shuffle=False` for `"test"`.

### 3.2 Base Detector (`src/detectors/base.py`)
```python
import torch
from torch import Tensor, nn

class BaseDetector(nn.Module):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input image tensor of shape [B, 3, H, W], float32, CLIP-normalized.
        Returns:
            score: Image-level anomaly score tensor of shape [B], float32 (higher = more anomalous).
            anomaly_map: Pixel-level anomaly map tensor of shape [B, H, W], float32.
        """
        raise NotImplementedError

    def set_reference(self, normal_images: Tensor | None) -> None:
        """Hook for Week 3 few-shot reference images. Zero-shot: no-op or raise NotImplementedError."""
        pass
```

### 3.3 Detection Metrics (`src/metrics/detection.py`)
All metric functions accept NumPy arrays:
- `image_auroc(labels: np.ndarray, scores: np.ndarray) -> float` (shapes `[N]`, `[N]`)
- `pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float` (shapes `[N, H, W]`, `[N, H, W]`)
- `pro_score(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, num_thresholds: int = 200) -> float`
- `evaluate_category(labels: np.ndarray, scores: np.ndarray, masks: np.ndarray, maps: np.ndarray) -> dict[str, float]`
  - Returns keys: `{"image_auroc": float, "pixel_auroc": float, "pro": float}`.

---

## 4. REQUIREMENT FOR THE `weekly_progress/` FOLDER

The user requires a dedicated folder (`weekly_progress/`) that tracks every week's tasks and explains the implementation so clearly that **even a beginner can understand it**.

### Structure of Each Weekly Guide
Whenever you complete a step in Section 5, update the corresponding file in `weekly_progress/`:
1. **`weekly_progress/README.md`**:
   - Project big picture: What is Industrial Anomaly Detection? What is a Vision-Language Model (CLIP)? Why do we need Explainable AI (XAI)?
   - Team 12 Role Split table (Prabhjot Singh: Data, Metrics, Differentiable Detector, IG/ABIGX; Sarhan Kapadiya: Env, WinCLIP Zero-Shot, AnomalyCLIP, Grad-CAM, VLM Explainer).
   - Quick-start commands for both **macOS** and **Windows**.
2. **`weekly_progress/glossary_for_beginners.md`**:
   - Plain-English definitions with simple analogies for: *Zero-Shot Learning, Embeddings, Cosine Similarity, Harmonic Mean, Sliding Windows (`unfold`), AUROC, Per-Region Overlap (PRO), Backpropagation / Input Gradients, MPS vs. CUDA*.
3. **`weekly_progress/week1_foundation_and_env.md`** (Covers Steps 1, 2, 3):
   - For each step:
     - **What We Built & Files Created**
     - **Why We Needed It (The Big Picture)**
     - **How It Works Under the Hood (Beginner Explanation)**
     - **Key Engineering Decisions (Mac & Windows compatibility, Week 4 gradient preparation)**
     - **How to Run & Verify (Commands + Expected Output)**
4. **`weekly_progress/week2_data_and_winclip.md`** (Covers Steps 4, 5, 6, 7, 8, 9, 10):
   - Same 5-part beginner breakdown for the MVTec AD dataset loader, Compositional Prompt Ensembling, Global Scoring, Multi-Scale Window Scoring, Gradient Flow verification, the Zero-Shot Runner, and the Checkpoint Comparison (`laion400m_e31` vs `laion400m_e32`).

---

## 5. STEP-BY-STEP BUILD PLAN & ACCEPTANCE TESTS

Execute these 10 steps **one at a time**. After each step, run `pytest` or the verification script, update `weekly_progress/`, and present the output.

---

### STEP 1: Utilities, Configuration, and Beginner Tracking Setup
**Goal:** Build cross-platform device selection, deterministic seeding, typed YAML config loading, and initialize the `weekly_progress/` folder.
- **Files to Create:**
  - `src/utils/device.py`: `get_device(forced_device=None)`. Auto-selects `cuda` $\rightarrow$ `mps` $\rightarrow$ `cpu`. Sets `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` when `mps` is chosen.
  - `src/utils/seed.py`: `seed_everything(seed=0)`. Seeds `random`, `numpy`, `torch`, `torch.cuda`, and `torch.mps`.
  - `src/utils/config.py`: `ZeroShotConfig` dataclass, `load_config()`, `save_config()` with UTF-8 encoding and `pathlib.Path` properties. Supports `score_mode` in `{"global", "map_max", "combined", "combined_smooth"}`.
  - `configs/zeroshot.yaml`: Default parameters (`image_size: 240`, `model: ViT-B-16-plus-240`, `pretrained: laion400m_e32`, `window_scales: [2, 3]`, `window_chunk_size: 16`, `num_workers: 0`, `seed: 0`).
  - `tests/test_utils_config.py`: Tests config loading, seed reproducibility (`torch.rand`), and device resolution.
  - `weekly_progress/README.md`, `weekly_progress/glossary_for_beginners.md`, and Step 1 section of `weekly_progress/week1_foundation_and_env.md`.
- **Acceptance Test:** Run `pytest tests/test_utils_config.py -v`. Must pass 100%.

---

### STEP 2: Environment & Differentiability Check (`scripts/check_env.py`)
**Goal:** Prove on Day 1 that `open_clip` loads `ViT-B-16-plus-240` and that gradients flow cleanly back to an input image on the host machine (Mac MPS/CPU or Windows CUDA/CPU).
- **Files to Create:**
  - `scripts/check_env.py`:
    1. Accepts `--device {auto,mps,cpu,cuda}` and `--config configs/zeroshot.yaml`.
    2. Prints Python version, OS platform, PyTorch version, available devices (`cuda`, `mps`, `cpu`), and all pretrained tags `open_clip` lists for `ViT-B-16-plus-240` (verifying `laion400m_e31` and `laion400m_e32` exist).
    3. Loads `ViT-B-16-plus-240` with the tag from config (`laion400m_e32`) in `float32`, freezes weights (`requires_grad_(False)`), and puts it in `.eval()` mode.
    4. Creates a random input tensor `x = torch.randn(1, 3, 240, 240, device=device, dtype=torch.float32, requires_grad=True)`.
    5. Runs `features = model.encode_image(x)`, computes `loss = features.sum()`, and calls `loss.backward()`.
    6. Asserts `x.grad is not None`, `torch.isfinite(x.grad).all()`, and `x.grad.abs().sum() > 0`.
    7. Exits with code `0` on success, `1` on any failure.
  - Update `weekly_progress/week1_foundation_and_env.md` with Step 2 explanation.
- **Acceptance Test:** Run `python scripts/check_env.py` on the machine and show full console output.

---

### STEP 3: Detection Metrics (`src/metrics/detection.py`)
**Goal:** Implement Image AUROC, Pixel AUROC, and Per-Region Overlap (PRO) with comprehensive synthetic unit tests (no dataset download needed yet).
- **Files to Create:**
  - `src/metrics/detection.py`:
    - `image_auroc(labels, scores)`: Uses `sklearn.metrics.roc_auc_score`. Returns `float("nan")` if only one class is present in `labels`.
    - `pixel_auroc(masks, maps)`: Flattens arrays in `float32` and computes `roc_auc_score`. Returns `float("nan")` if all mask pixels are 0 or all are 1, or `0.5` if `maps` is constant.
    - `pro_score(masks, maps, fpr_limit=0.3, num_thresholds=200)`:
      - Uses `scipy.ndimage.label` on each 2D ground-truth mask to find connected anomalous components.
      - Computes `num_thresholds` linearly spaced between `maps.min()` and `maps.max()`.
      - At each threshold, computes the mean per-component recall (overlap pixels / component area) across all components in the category, and the False Positive Rate (FPR) over all normal pixels (`masks == 0`).
      - Sorts by FPR, clamps/interpolates at `fpr_limit = 0.3`, integrates PRO vs. FPR from `0` to `0.3` using `np.trapezoid` (or `scipy.integrate.trapezoid`), and divides by `0.3` so a perfect detector scores `1.0`.
      - Handles edge cases gracefully: constant maps return `0.0`, all-zero masks return `float("nan")`.
  - `tests/test_metrics.py`:
    1. Random scores (fixed seed) yield Image and Pixel AUROC in `[0.45, 0.55]`.
    2. Perfect anomaly map (`maps == masks`) yields Pixel AUROC `1.0` and PRO `> 0.99`.
    3. Inverted anomaly map (`maps == 1.0 - masks`) yields AUROC `< 0.1`.
    4. Hand-built toy example with two small ground-truth regions (e.g., a $2\times2$ square of area 4 and a $1\times4$ rectangle of area 4 inside a $10\times10$ image) with known threshold overlaps; include the exact hand-calculated PRO math in code comments and assert agreement within `1e-2`.
    5. Constant maps and empty masks do not crash.
  - Update `weekly_progress/week1_foundation_and_env.md` with Step 3 explanation (including a visual diagram in ASCII/Markdown showing how AUROC vs. PRO works).
- **Acceptance Test:** Run `pytest tests/test_metrics.py -v`.

---

### STEP 4: MVTec AD Dataset Loader (`src/data/mvtec.py`)
**Goal:** Load all 15 MVTec AD categories with strict tensor shapes, nearest-neighbor binary masks, and cross-platform path support.
- **Files to Create:**
  - `src/data/mvtec.py`:
    - Constant `MVTEC_CATEGORIES` listing all 15 categories: `bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper`.
    - `MVTecDataset(root, category, split="test", image_size=240)`:
      - Checks if `Path(root) / category` exists; if missing, raises `FileNotFoundError` with clear instructions on where to download MVTec AD and how to extract it into `data/mvtec_ad/`.
      - Matches every `test/<defect_type>/<id>.png` with `ground_truth/<defect_type>/<id>_mask.png` (except `defect_type == "good"`, which gets an all-zero mask `[1, 240, 240]`).
      - Resizes RGB images to `(240, 240)` using bicubic interpolation (`InterpolationMode.BICUBIC`), converts to `[0, 1]` float32 tensor, and normalizes with CLIP mean and std **exactly once**.
      - Resizes masks to `(240, 240)` using `InterpolationMode.NEAREST` and binarizes at `> 0.5` into `{0.0, 1.0}` float32 tensor of shape `[1, 240, 240]`.
  - `tests/test_dataset.py`:
    - Includes a synthetic dummy MVTec directory fixture (`tmp_path`) so unit tests always verify contract shapes/dtypes even before the 5 GB dataset is downloaded.
    - Also includes a real dataset test (auto-skipped via `pytest.mark.skipif` only if `data/mvtec_ad` is not yet present on disk) that checks `carpet` test count (expected: 117 test images) and prints category counts.
  - Update `weekly_progress/week2_data_and_winclip.md` with Step 4 explanation.
- **Acceptance Test:** Run `pytest tests/test_dataset.py -v`.

---

### STEP 5: Compositional Prompts & Detector Skeleton (`src/detectors/base.py`, `prompts.py`, `winclip.py`)
**Goal:** Build WinCLIP's Compositional Prompt Ensemble (CPE) and the `BaseDetector` / `WinCLIPDetector` skeleton.
- **Files to Create:**
  - `src/detectors/base.py`: `BaseDetector` interface as specified in Section 3.2.
  - `src/detectors/prompts.py`:
    - Normal state templates (`"{}"`, `"flawless {}"`, `"perfect {}"`, `"unblemished {}"`, `"{} without flaw"`, `"{} without defect"`, `"{} without damage"`) — mark `# VERIFY against WinCLIP paper Section 3.1`.
    - Anomalous state templates (`"damaged {}"`, `"{} with flaw"`, `"{} with defect"`, `"{} with damage"`, `"broken {}"`) — mark `# VERIFY against WinCLIP paper Section 3.1`.
    - Photo templates (`"a cropped photo of the {}."`, `"a close-up photo of a {}."`, `"a photo of a {} for visual inspection."`, etc.).
    - Category display names mapping (`"metal_nut": "metal nut"`, `"grid": "grid"`, etc.).
    - `build_text_embedding(model, tokenizer, category, device, cache_dir=None, pretrained_tag="") -> tuple[Tensor, dict]`:
      - Encodes all normal prompts, L2-normalizes each, averages them, and L2-normalizes the mean vector $\rightarrow \mathbf{t}_{\text{normal}} \in \mathbb{R}^{1 \times D}$.
      - Repeats for anomalous prompts $\rightarrow \mathbf{t}_{\text{anomalous}} \in \mathbb{R}^{1 \times D}$.
      - Stacks into `[2, D]` unit-norm tensor (Row 0 = Normal, Row 1 = Anomalous).
      - Saves/loads from `results/cache/text_emb_<category>_<pretrained_tag>.pt`.
  - `src/detectors/winclip.py`:
    - Initializes `open_clip` model, freezes all parameters (`param.requires_grad_(False)`), and stores the `[2, D]` text matrix.
  - `docs/open_questions.md`: Log all `# VERIFY` items (exact prompt lists, window sizes, stride, harmonic mean vs. arithmetic mean across scales, Gaussian smoothing $\sigma$).
  - `tests/test_winclip_shapes.py`: Asserts `[2, D]` text embedding is unit-norm and detector returns `(score [B], map [B, 240, 240])`.
  - Update `weekly_progress/week2_data_and_winclip.md` with Step 5 explanation.
- **Acceptance Test:** Run `pytest tests/test_winclip_shapes.py -v`.

---

### STEP 6: WinCLIP Stage (a) — Global Image Scoring
**Goal:** Implement global CLIP image-text scoring inside `WinCLIPDetector.forward`.
- **Implementation Details:**
  - Encode full image $x \in \mathbb{R}^{B \times 3 \times 240 \times 240}$ via `model.encode_image(x)`.
  - Normalize image embedding: $\hat{\mathbf{f}} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$.
  - Compute logits: $\text{logits} = \exp(\text{logit\_scale}) \cdot (\hat{\mathbf{f}} @ \mathbf{T}^\top) \in \mathbb{R}^{B \times 2}$ (where $\exp(\text{logit\_scale})$ is CLIP's learned temperature, ~100.0).
  - Apply `F.softmax(logits, dim=-1)` and take column `1` (`probs[:, 1]`) as the global anomaly probability $s_{\text{global}} \in [0, 1]$.
  - When `score_mode == "global"`, return `(s_global, s_global[:, None, None].expand(B, H, W))`.
- **Acceptance Test:** Run `pytest tests/test_winclip_shapes.py -v` and verify global scores on a sample batch. Update `weekly_progress/week2_data_and_winclip.md`.

---

### STEP 7: WinCLIP Stage (b) — Multi-Scale Window Scoring & Aggregation
**Goal:** Implement differentiable sliding-window multi-scale localization inside `WinCLIPDetector.forward`.
- **Implementation Details (`src/detectors/winclip.py`):**
  - Base ViT patch size for `ViT-B-16-plus-240` is $16 \times 16$ pixels (yielding a $15 \times 15$ grid for a $240 \times 240$ image).
  - For each scale $k \in \text{window\_scales}$ (default `[2, 3]`, corresponding to $2\times2$ patch windows of size $32\times32$ and $3\times3$ patch windows of size $48\times48$ with stride $16$ — `# VERIFY against WinCLIP paper Section 3.2`):
    1. Extract sliding windows from input tensor $x$ using `F.unfold(x, kernel_size=win_size, stride=stride)`.
    2. Reshape windows into `[B * N_win, 3, win_size, win_size]`.
    3. Process windows in chunks of `window_chunk_size` (default `16`):
       - Resize each chunk to `(240, 240)` via `F.interpolate(..., mode="bicubic", align_corners=False)`.
       - Encode via `model.encode_image` (with optional `torch.utils.checkpoint.checkpoint` when `use_checkpointing=True` and `x.requires_grad` is True).
       - Compute anomalous probability $S_w \in (0, 1)$ against the `[2, D]` text matrix.
    4. **Numerically Stable Harmonic Mean Aggregation:**
       - Clamp window scores: $S_w^{\text{safe}} = \text{ torch.clamp}(S_w, \min=10^{-6})$.
       - Compute reciprocal $1.0 / S_w^{\text{safe}}$ and use `F.fold` (with a ones-mask folded the same way to count overlapping windows $K(i,j)$ per pixel) to compute the exact pixel-wise harmonic mean at scale $k$:
         $$M_k(i,j) = \frac{K_k(i,j)}{\sum_{w \ni (i,j)} \frac{1}{S_w^{\text{safe}}}}$$
  - Average the harmonic-mean maps across all scales $k$: $M = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} M_k$.
  - Optionally apply differentiable Gaussian smoothing (`smooth_sigma > 0`) using a fixed 2D Gaussian conv kernel in PyTorch.
  - Compute final image score based on `score_mode`:
    - `"global"`: $s_{\text{global}}$
    - `"map_max"`: $M.\text{amax}(\text{dim}=(1,2))$
    - `"combined"`: $0.5 \cdot (s_{\text{global}} + M.\text{amax}(\text{dim}=(1,2)))$
    - `"combined_smooth"`: $0.5 \cdot \left(s_{\text{global}} + \frac{1}{\tau}\text{torch.logsumexp}(\tau \cdot M.\text{flatten}(1), \text{dim}=1)\right)$ with $\tau = 20.0$.
- **Acceptance Test:** Measure memory and runtime per image for batch size 1, and update `weekly_progress/week2_data_and_winclip.md`.

---

### STEP 8: End-to-End Gradient Flow & Cross-Platform Device Consistency Tests
**Goal:** Prove that gradients flow through the entire multi-scale WinCLIP pipeline (`unfold` $\rightarrow$ `interpolate` $\rightarrow$ `ViT` $\rightarrow$ `harmonic mean fold`) and that Mac MPS / Windows CUDA match CPU within `1e-3`.
- **Files to Create:**
  - `tests/test_grad_flow.py`:
    - Instantiates `WinCLIPDetector` (tested across `"global"`, `"combined"`, and `"combined_smooth"` score modes with a small test configuration).
    - Passes a batch of 2 random images `x = torch.randn(2, 3, 240, 240, requires_grad=True)`.
    - Calls `scores, maps = detector(x)` and `scores.sum().backward()`.
    - Asserts `x.grad is not None`, `x.grad.shape == x.shape`, `torch.isfinite(x.grad).all()`, and `x.grad.abs().sum() > 0`.
  - `tests/test_device_consistency.py`:
    - Checks if an accelerator (`mps` on Mac or `cuda` on Windows/Linux) is available; skips via `pytest.skip` if only `cpu` is available.
    - Scores 4 deterministic synthetic images on `cpu` and on the accelerator (`mps` or `cuda`).
    - Asserts `torch.allclose(scores_cpu, scores_acc.cpu(), atol=1e-3)`.
  - Update `weekly_progress/week2_data_and_winclip.md` with Step 8 explanation.
- **Acceptance Test:** Run `pytest tests/test_grad_flow.py tests/test_device_consistency.py -v`.

---

### STEP 9: Zero-Shot Evaluation Runner (`scripts/run_zeroshot.py`)
**Goal:** Build the production CLI runner with caching, resumable category execution, per-image `.npz` saving, and `metrics.csv` logging.
- **Files to Create:**
  - `scripts/run_zeroshot.py`:
    - CLI flags: `--config configs/zeroshot.yaml`, `--category <name|all>`, `--device {auto,mps,cpu,cuda}`, `--max_images <int|None>`, `--overwrite`.
    - Runs inference inside `with torch.no_grad():`.
    - Saves under `results/<run_name>/<category>/`:
      - `predictions.npz`: `scores` (`float32`), `maps` (`float16`), `labels` (`int32`), `paths` (POSIX strings), `defect_types` (strings).
      - `config_used.yaml` and `prompts_used.json`.
    - Appends/updates `results/<run_name>/metrics.csv` with columns:
      `category,image_auroc,pixel_auroc,pro,num_images,sec_per_image,device,pretrained,score_mode`.
    - Skips categories that already have `predictions.npz` unless `--overwrite` is passed.
  - Update `weekly_progress/week2_data_and_winclip.md` with Step 9 explanation.
- **Acceptance Test:** Run `python scripts/run_zeroshot.py --config configs/zeroshot.yaml --category carpet --max_images 20` (if MVTec AD is present, or on a synthetic test harness if not yet downloaded) and display the generated CSV row.

---

### STEP 10: Checkpoint Comparison (`laion400m_e31` vs `laion400m_e32`) & Sanity Check
**Goal:** Compare the two OpenCLIP LAION-400M checkpoints and compare zero-shot metrics against the WinCLIP CVPR 2023 paper.
- **Files to Create:**
  - `scripts/compare_checkpoints.py`:
    - Runs `laion400m_e31` and `laion400m_e32` on the same subset (`--max_images 20` on `carpet` and `leather`) and writes a side-by-side comparison markdown/CSV table to `results/checkpoint_comparison.md`.
  - `docs/open_questions.md`:
    - Record the WinCLIP CVPR 2023 paper's reported zero-shot numbers for MVTec AD (Paper Table 1 / Supplement: MVTec-AD Zero-Shot mean Image AUROC is **91.8%**, Pixel AUROC is **85.1%**, and PRO is **64.6%**).
    - Document any discrepancies and how prompt lists, window scales, or Gaussian smoothing affect the gap.
  - Finalize `weekly_progress/week2_data_and_winclip.md`.
- **Acceptance Test:** Run `pytest` across the entire test suite (`pytest -v`) and verify all tests pass.