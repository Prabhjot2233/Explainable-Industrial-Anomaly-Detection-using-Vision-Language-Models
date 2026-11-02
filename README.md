# Explainable Industrial Anomaly Detection using Vision-Language Models

COMP 8977 – Team 12. Zero-shot and few-shot anomaly detection on industrial images (MVTec AD)
using CLIP-style vision-language models, with visual explanations of *why* an image is flagged
(Saliency, Integrated Gradients, ViT Grad-CAM).

## Folder structure

| Folder | What lives here |
|---|---|
| `src/` | ML pipeline code (importable as `src.*`) |
| `src/utils/` | Shared helpers: `device.py` (cuda/mps/cpu), `seed.py` (reproducibility), `config.py` (YAML config) |
| `src/data/` | `mvtec.py`: MVTec AD loader (240×240, CLIP-normalised, masks) |
| `src/detectors/` | `winclip.py` (zero/few-shot WinCLIP+), `patchcore.py`, `anomalyclip.py`, `prompts.py`, `base.py` |
| `src/metrics/` | `detection.py`: image AUROC, pixel AUROC, PRO |
| `src/explainers/` | `saliency.py`, `ig.py` (IG + Expected-IG), `gradcam_vit.py`, `visualize.py` |
| `configs/` | Experiment configs: `zeroshot.yaml`, `fewshot.yaml`, `baselines.yaml`, `explanations_w4.yaml` |
| `scripts/` | Runners: `run_fewshot.py`, `run_baselines.py`, `run_visual_explanations.py` |
| `tests/` | Pytest unit tests for `src/` (run from the repo root) |
| `backend/` | FastAPI server exposing the model (image upload, detection API) |
| `frontend/` | React + Vite web UI |
| `docs/` | Implementation plan and project documentation |
| `research_papers/` | Reference papers |
| `proposal/` | Team 12 project proposal |
| `data/` | Datasets, e.g. MVTec AD (git-ignored) |
| `results/` | Predictions, metrics, caches (git-ignored) |
| `weekly_progress/` | **Beginner guides: how the project works + weekly tasks/progress** |

## How to run it

> **Important:** run every ML command from the **repo root** (the folder containing this
> README), *not* from `backend/`. Running `pytest` inside `backend/` gives
> `ERROR: file or directory not found: tests`.

### Step 1: Set up the environment (one time)

**macOS**
```bash
cd "internship projet II"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd "internship projet II"
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> In each new terminal, re-activate the venv (`source .venv/bin/activate` or
> `.venv\Scripts\Activate.ps1`). If `python` isn't found, you forgot this step.

### Step 2: Run the tests (no dataset or downloads needed)

```bash
python -m pytest tests -v
```
Expected: `36 passed`. The tests use a tiny random CLIP model, so they run offline in about 15 s.

Run a single step's tests:
```bash
python -m pytest tests/test_fewshot.py -v          # Step 11
python -m pytest tests/test_patchcore.py -v        # Step 12
python -m pytest tests/test_anomalyclip.py -v      # Step 13
python -m pytest tests/test_saliency_ig.py -v      # Steps 14-15
python -m pytest tests/test_gradcam_visualize.py -v  # Steps 16-17
```

### Step 3: Download the dataset (one time, ~5 GB)

Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it so that:
```
data/mvtec_ad/bottle/train/good/*.png
data/mvtec_ad/bottle/test/<defect>/*.png
data/mvtec_ad/bottle/ground_truth/<defect>/*_mask.png
```

### Step 4: Run the experiments

The first run downloads model weights: CLIP ViT-B-16-plus-240 (~800 MB) and WideResNet50 (~130 MB).
Start with `--max_images` for a quick check, then remove it for the full run.

```bash
# Week 3 / Step 11: few-shot WinCLIP+ (k = 0,1,2,4) + prompt ablation
python scripts/run_fewshot.py --category bottle --k_shots all --prompt_mode all --max_images 20

# Week 3 / Step 13: compare WinCLIP zero/4-shot, AnomalyCLIP, PatchCore
python scripts/run_baselines.py --detector all --category bottle --max_images 20

# Week 4 / Step 17: Saliency, ViT Grad-CAM, IG (black), Expected-IG heatmaps
python scripts/run_visual_explanations.py --category bottle --max_images 3
```

| Flag | Meaning |
|---|---|
| `--category <name\|all>` | One MVTec category, or all categories in the config |
| `--max_images N` | Evaluate only N test images (omit for the full test set) |
| `--device auto\|cpu\|mps\|cuda` | Hardware (default `auto`: cuda → mps → cpu) |
| `--k_shots`, `--prompt_mode` | Few-shot only: e.g. `--k_shots 2 --prompt_mode single` |
| `--detector` | Baselines only: `all`, `winclip`, `anomalyclip`, `patchcore` |
| `--overwrite` | Few-shot only: recompute rows already in the CSV |

Settings (categories, window scales, IG steps, etc.) live in `configs/*.yaml`.

### Step 5: Find the results

| Output | Location |
|---|---|
| Few-shot metrics | `results/fewshot/fewshot_metrics.csv`, `prompt_ablation.csv` |
| Detector comparison | `results/baselines/detector_comparison.csv` |
| Explanation grids + maps | `results/explanations_w4/<category>/*.png`, `*.npz` |
| Explainer runtimes | `results/explanations_w4/runtime_summary.csv` |

> **Speed tip:** WinCLIP encodes ~365 windows per image, so CPU runs are slow. Apple Silicon
> uses `mps` and NVIDIA uses `cuda` automatically. Expected-IG is the slowest explainer, so
> keep `--max_images` small.

### Step 6 (optional): Run the web app

```bash
# Terminal 1: backend (venv activated)
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload        # API docs: http://127.0.0.1:8000/docs

# Terminal 2: frontend
cd frontend
npm install
npm run dev                          # UI: http://localhost:5173
```

## Plan

- Technical plans: [Weeks 1–2](docs/implementation_plan_weeks_1_2.md), [Weeks 3–4](docs/master_implementation_plan_weeks_3_4.md)
- Beginner guides and progress: [weekly_progress/README.md](weekly_progress/README.md)
  ([Week 3](weekly_progress/week3_detection_baselines.md), [Week 4](weekly_progress/week4_visual_explanations.md))
