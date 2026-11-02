# Week 3: Few-Shot Detection and Baselines (Steps 11–13)

| Step | Task | Key files | Status |
|---|---|---|---|
| 11 | Few-shot WinCLIP+ (k = 1, 2, 4) and prompt ablation | `src/detectors/prompts.py`, `winclip.py`, `scripts/run_fewshot.py` | ✅ |
| 12 | Differentiable PatchCore baseline | `src/detectors/patchcore.py` | ✅ |
| 13 | AnomalyCLIP detector and baselines runner | `src/detectors/anomalyclip.py`, `scripts/run_baselines.py` | ✅ |

> **Prerequisite added this week:** the Weeks 1–2 foundations (`src/data/mvtec.py`,
> `src/detectors/base.py`, `src/metrics/detection.py`, zero-shot WinCLIP) were built at the
> same time, because Week 3 builds directly on top of them.

---

## Step 11: Few-shot WinCLIP+ and prompt ablation

### 1. Plain-English analogy
**Zero-shot** is an inspector who has never seen your carpet. They only know the general idea
of "damaged" from words. **Few-shot** means giving them 1, 2 or 4 photos of a brand-new carpet
before their shift starts. Now they can also ask: *"does this patch look like anything in my
reference photos?"* If it doesn't, it's suspicious.

**Prompt ablation** tests how much the wording matters: one short sentence, a handful of
"state" words, or the full list of templates × states (the full ensemble).

### 2. How the math works
1. Every window of the test image becomes a direction (an *embedding*) of length 1 in
   CLIP's meaning space.
2. The same is done once for every window of the k normal photos. That's the reference bank `R`.
3. **Cosine similarity** `F @ Rᵀ` scores how alike two windows are (1 = identical direction).
4. For each test window, keep its **best match** `max` (or a smooth max, `logsumexp`, when we
   need clean gradients). Its anomaly score is `½(1 − best match)`: a close match gives a score
   near 0, no good match gives a higher score.
5. The window scores are spread back onto pixels with a **harmonic mean**, which gives the
   visual heatmap `M_vis`.
6. Final map = `0.5 × text map + 0.5 × visual map`.

Prompt modes:
- `single`: "a photo of a bottle." vs "a photo of a damaged bottle." (1 sentence each)
- `state_only`: 1 template × all state words (7 normal, 5 anomalous)
- `full_ensemble`: 13 templates × all states (the default)

### 3. Files and functions
- `src/detectors/prompts.py`: `build_prompts()` creates the sentences for each mode.
  `build_text_embedding()` averages their CLIP embeddings and caches them as
  `text_emb_<category>_<tag>_<prompt_mode>.pt`.
- `src/detectors/winclip.py`
  - `set_reference(imgs)` stores the normal window embeddings. `set_reference(None)` switches back to zero-shot.
  - `forward(x)` encodes each window **once** and uses it for both the text score and the reference score.
- `configs/fewshot.yaml`: categories, `k_shots_list: [0,1,2,4]`, prompt modes.
- `scripts/run_fewshot.py`: writes `results/fewshot/fewshot_metrics.csv` and `prompt_ablation.csv`.
- `tests/test_fewshot.py`: checks embeddings are unit-length, that modes differ, that few-shot changes the map, and that gradients still flow.

### 4. How to run
macOS: `./.venv/bin/python -m pytest tests/test_fewshot.py -v`
Windows: `.venv\Scripts\python -m pytest tests/test_fewshot.py -v`

Full experiment (needs MVTec in `data/mvtec_ad/`, downloads CLIP weights on first run):
```bash
python scripts/run_fewshot.py --category bottle --k_shots all --prompt_mode all --max_images 30
```
Expected: one printed dict per (category, k, mode), e.g.
`{'category': 'bottle', 'k_shots': 2, 'prompt_mode': 'full_ensemble', 'image_auroc': ..., ...}`

---

## Step 12: Differentiable PatchCore

### 1. Plain-English analogy
PatchCore is like a **scrapbook of normal texture snippets**. For every small patch of a new
image, you flip through the scrapbook and find the closest snippet. If even the closest one is
far away, that patch is abnormal. No language is involved. It is the classic "non-VLM" baseline.

### 2. How the math works
1. Images are converted from CLIP colour normalisation back to `[0,1]`, then re-normalised the
   way ImageNet networks expect. This is plain arithmetic, so gradients survive.
2. A WideResNet-50 extracts `layer2` (30×30 grid) and `layer3` (15×15, upsampled to 30×30).
   Each is softened with a 3×3 average, then they are joined into a 1536-number vector per patch and normalised to length 1.
3. **Memory bank:** all patch vectors from the k normal images, randomly subsampled with a fixed
   seed (`coreset_ratio`, at most 2000 vectors).
4. For unit vectors, distance² = `2 − 2·cosine`. The minimum distance per patch gives a 30×30 map, which is upsampled to 240×240 and smoothed with a Gaussian (σ = 4).
5. Image score = maximum of the map (a smooth maximum in `combined_smooth` mode).

### 3. Files
- `src/detectors/patchcore.py`: `PatchCoreDetector` with `_features`, `set_reference`, `forward`.
- `tests/test_patchcore.py`: checks the memory bank size, output shapes, input gradients, and that a defective square scores higher.

### 4. How to run
`python -m pytest tests/test_patchcore.py -v` (the test uses random weights so it stays offline;
the real runner downloads ImageNet weights, about 130 MB, the first time).

---

## Step 13: AnomalyCLIP and the baselines runner

### 1. Plain-English analogy
WinCLIP asks about "a damaged **bottle**". AnomalyCLIP asks about "a damaged **object**". It
uses one set of generic sentences for every product. Think of an inspector trained on *defects in
general* rather than on one product.

### 2. How the math works
- Text: average embedding of 2 "normal object" and 2 "damaged object" sentences, plus an optional
  offset `ΔT` (starts at zero, can be trained later).
- Image: instead of sliding windows, it reads the ViT's 15×15 **patch tokens** directly and
  projects them into text space.
- Each patch: `softmax(100 · token · textᵀ)` gives P(anomalous). The 15×15 grid is upsampled to 240×240 and smoothed.

### 3. Files
- `src/detectors/anomalyclip.py`: `AnomalyCLIPDetector`.
- `configs/baselines.yaml`: `winclip_zeroshot`, `winclip_4shot`, `anomalyclip`, `patchcore`.
- `scripts/run_baselines.py`: writes `results/baselines/detector_comparison.csv`
  (`detector,category,k_shots,image_auroc,pixel_auroc,pro,sec_per_image,device`).
- `tests/test_anomalyclip.py`: checks shapes, a `[0,1]` map, and finite gradients.

### 4. How to run
```bash
python -m pytest tests/test_anomalyclip.py -v
python scripts/run_baselines.py --detector all --category bottle --max_images 30
```
On Windows use `.venv\Scripts\python` instead of `python` if the venv isn't activated.
