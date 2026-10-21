# Explainable Industrial Anomaly Detection using Vision-Language Models

COMP 8977 – Team 12. Zero-shot anomaly detection on industrial images (MVTec AD) using
CLIP-style vision-language models, with explanations of *why* an image is flagged.

## Folder structure

| Folder | What lives here |
|---|---|
| `src/` | ML pipeline code (importable as `src.*`) |
| `src/utils/` | Shared helpers: `device.py` (cuda/mps/cpu), `seed.py` (reproducibility), `config.py` (YAML config) |
| `configs/` | Experiment config files (e.g. `zeroshot.yaml`) |
| `scripts/` | Runnable entry points (env check, evaluation, etc.) |
| `tests/` | Pytest unit tests for `src/` |
| `backend/` | FastAPI server exposing the model (image upload, detection API) |
| `frontend/` | React + Vite web UI |
| `docs/` | Implementation plan and project documentation |
| `research_papers/` | Reference papers |
| `proposal/` | Team 12 project proposal |
| `data/` | Datasets, e.g. MVTec AD (git-ignored) |
| `results/` | Predictions, metrics, caches (git-ignored) |
| `weekly_progress/` | **Beginner guides: how the project works + weekly tasks/progress** |

## Quick start (ML pipeline)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -v
```

## Web app

```bash
# Terminal 1: backend (venv activated, from repo root)
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload        # http://127.0.0.1:8000/docs

# Terminal 2: frontend (from repo root)
cd frontend
npm install
npm run dev                          # http://localhost:5173
```

## Plan

Technical plan: [docs/implementation_plan_weeks_1_2.md](docs/implementation_plan_weeks_1_2.md) — beginner guides and weekly progress: [weekly_progress/README.md](weekly_progress/README.md).
