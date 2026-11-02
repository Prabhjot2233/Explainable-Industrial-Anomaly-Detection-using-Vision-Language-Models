# Week 4: Visual Explanations (Steps 14–17)

| Step | Task | Key files | Status |
|---|---|---|---|
| 14 | Base explainer, score wrapper, Saliency | `src/explainers/base.py`, `saliency.py` | ✅ |
| 15 | Integrated Gradients (black and Expected-IG) | `src/explainers/ig.py` | ✅ |
| 16 | ViT Grad-CAM | `src/explainers/gradcam_vit.py` | ✅ |
| 17 | Heatmap visualisation and Week 4 runner | `src/explainers/visualize.py`, `scripts/run_visual_explanations.py` | ✅ |

---

## Step 14: Score wrapper and Saliency

### 1. Plain-English analogy
Saliency asks: *"if I nudged this pixel a tiny bit, how much would the anomaly score change?"*
Pixels the score is sensitive to light up. It's like pressing each key on a piano very gently
and listening for which ones make the loudest sound.

### 2. How the math works
- Detectors return `(score, map)`, but explainers need **one number per image**.
  `DifferentiableScoreWrapper` returns only the score and temporarily switches to
  `combined_smooth` mode. A hard `max` only passes gradient through one pixel, while `logsumexp`
  spreads it smoothly. The original mode is restored afterwards (in a `finally` block).
- Saliency = ∂score / ∂pixel (the gradient). For each pixel we take the length of the gradient
  across the 3 colours, then rescale the whole map to `[0, 1]`.

### 3. Files
- `src/explainers/base.py`: `DifferentiableScoreWrapper`, `BaseExplainer.aggregate_and_normalize`.
- `src/explainers/saliency.py`: `SaliencyExplainer` (Captum `Saliency`).

### 4. Run
`python -m pytest tests/test_saliency_ig.py -v -k "saliency or wrapper"`

---

## Step 15: Integrated Gradients

### 1. Plain-English analogy
Imagine a **dimmer switch** going from a dark room (the *baseline*) to full brightness (the real
image). At each step you check how much the inspector's suspicion grows and which pixels caused
it. Add up all those small contributions and you get each pixel's share of the final score.

### 2. How the math works
`IG_i = (x_i − baseline_i) × average over α∈[0,1] of ∂score/∂x_i at (baseline + α(x − baseline))`

- **Completeness axiom:** the attributions add up to `score(image) − score(baseline)`. We measure
  the gap as `meta["completeness_delta"]`, and the test requires it to be below 0.05.
- **Black baseline:** an all-zero tensor, which in CLIP-normalised space is the *average colour*
  (`true_black` = genuine RGB black).
- **Expected-IG:** instead of a dark room, start from each of K **normal** photos and average
  the results. The question becomes "what makes this image different from a *normal* one?",
  which suits anomaly detection better.
- `n_steps` = dimmer positions; `internal_batch_size=4` keeps GPU/MPS memory low.

### 3. Files
- `src/explainers/ig.py`: `IntegratedGradientsExplainer(baseline_mode="black"|"true_black"|"expected")`.

### 4. Run
`python -m pytest tests/test_saliency_ig.py -v`

---

## Step 16: ViT Grad-CAM

### 1. Plain-English analogy
The Vision Transformer cuts the image into a 15×15 grid of tiles. Grad-CAM looks inside the
**last layer** and asks: *"which tiles carry features that push the anomaly score up?"* It is
coarser than pixel methods, but fast (one forward and one backward pass).

### 2. How the math works
1. A forward hook records the activations `A` of the last transformer block (226 tokens = 1 [CLS] + 225 tiles).
   A tensor hook records their gradients `G`. The [CLS] token is dropped.
2. Channel weights `w = mean of G over tiles`.
3. `CAM = ReLU(Σ_d w_d · A_d)` gives a 15×15 grid. It is upsampled (bicubic) to 240×240 and rescaled to `[0, 1]`.
4. WinCLIP runs the encoder many times (whole image + every window). Only the **first,
   full-image** pass is captured. Hooks are always removed in `finally`.

### 3. Files
- `src/explainers/gradcam_vit.py`: `ViTGradCAMExplainer`.

### 4. Run
`python -m pytest tests/test_gradcam_visualize.py -v -k gradcam`

---

## Step 17: Visualisation and Week 4 runner

### 1. Plain-English analogy
A side-by-side "report card" for each defective image: the photo, the true defect mask, the
detector's heatmap, and what each explanation method thinks mattered.

### 2. Files and functions
- `src/explainers/visualize.py`
  - `denormalize_clip_image`: tensor → viewable RGB image.
  - `create_heatmap_overlay`: blends a `[0,1]` map onto the photo using the `jet` colour map.
  - `save_explanation_comparison_grid`: 7-panel PNG at 150 DPI (Agg backend, so no pop-up windows).
- `configs/explanations_w4.yaml` and `scripts/run_visual_explanations.py`: run all 4 explainers on
  defective test images and save `.npz` maps, `.png` grids and `runtime_summary.csv` under
  `results/explanations_w4/`.

### 3. How to run
```bash
python -m pytest tests/ -v
python scripts/run_visual_explanations.py --category bottle --max_images 5
```
Windows: same commands with `.venv\Scripts\python`. Expected output: one line per
(image, explainer) with its runtime, plus PNGs in `results/explanations_w4/bottle/`.
