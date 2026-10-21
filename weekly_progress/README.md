# Weekly Progress — Start Here

This folder explains **what we built, why, and how it works** in plain English. You don't need
any background in Python, PyTorch or computer vision to follow it.

## How to read this folder

| File | Read it when you want to… |
|---|---|
| `README.md` (this file) | Get the big picture and run the project |
| [`glossary_for_beginners.md`](glossary_for_beginners.md) | Look up a word you don't know (CLIP, AUROC, gradient…) |
| [`week1_foundation_and_env.md`](week1_foundation_and_env.md) | Understand Steps 1–3: utilities, environment check, metrics |
| [`week2_data_and_winclip.md`](week2_data_and_winclip.md) | Understand Steps 4–10: dataset, WinCLIP detector, evaluation |

## Progress tracker

| Step | Week | What | Status |
|---|---|---|---|
| 1 | 1 | Utilities: device, seed, config | ✅ Done (tests pass) |
| 2 | 1 | Environment and gradient check (`scripts/check_env.py`) | ⏳ Next |
| 3 | 1 | Detection metrics: AUROC and PRO | ⬜ |
| 4 | 2 | MVTec AD dataset loader | ⬜ |
| 5 | 2 | Prompts and detector skeleton | ⬜ |
| 6 | 2 | WinCLIP global scoring | ⬜ |
| 7 | 2 | WinCLIP multi-scale window scoring | ⬜ |
| 8 | 2 | Gradient flow and device consistency tests | ⬜ |
| 9 | 2 | Zero-shot evaluation runner | ⬜ |
| 10 | 2 | Checkpoint comparison (e31 vs e32) | ⬜ |

---

## The big picture

### What is industrial anomaly detection?
Factories photograph every product (a bottle, a carpet, a transistor) and need to catch the
faulty ones: scratches, holes, missing parts. Defects are **rare and varied**, so you can't
collect thousands of examples of every possible fault. We want a system that says:
*"this looks abnormal, and here is where."*

**Analogy:** a quality inspector who has never seen this exact product, but knows what
"broken", "scratched" or "damaged" look like in general.

### What is a vision-language model (CLIP)?
CLIP was trained on hundreds of millions of image + caption pairs from the internet. It learned
to place images and sentences in the **same "meaning space"**, so a photo of a cracked bottle
lands close to the sentence *"a photo of a damaged bottle"*.

That lets us do **zero-shot** detection: no training on defects. We simply ask CLIP whether
the image is closer to "flawless bottle" or "damaged bottle".

### What is WinCLIP?
CLIP looks at the whole image at once, so a tiny scratch can get lost. **WinCLIP** slides small
windows across the image, asks CLIP about each window, and combines the answers into a **heatmap**
showing where the defect is.

### Why explainable AI (XAI)?
A factory operator won't trust a black box that just says "reject". In later weeks (Week 4) we
explain each decision using **Integrated Gradients** and **ABIGX**, which trace the score back
to the exact pixels that caused it. That's why the detector must be written so **gradients can
flow back to the image** (see the glossary).

### Dataset: MVTec AD
The standard benchmark: 15 product categories (5 textures, 10 objects) with pixel-accurate
defect masks. We start with `carpet`, `leather`, `tile`, `bottle` and `transistor`.

---

## Team 12 role split

| Member | Responsibilities |
|---|---|
| **Prabhjot Singh** | Data loading, metrics, differentiable detector, Integrated Gradients / ABIGX |
| **Sarhan Kapadiya** | Environment, WinCLIP zero-shot, AnomalyCLIP, Grad-CAM, VLM explainer |

---

## Quick start

### macOS
```bash
cd "internship projet II"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -v
```

### Windows (PowerShell)
```powershell
cd "internship projet II"
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests -v
```

> If you see `command not found: python`, you forgot to activate the venv (the `source` /
> `Activate.ps1` line). You can also run `./.venv/bin/python -m pytest tests -v` directly.

## Where things live
- `src/`: the actual code (`utils/` now; `data/`, `detectors/` and `metrics/` coming)
- `configs/zeroshot.yaml`: all experiment settings in one place
- `scripts/`: programs you run (env check, evaluation)
- `tests/`: automatic checks that prove each step works
- `docs/implementation_plan_weeks_1_2.md`: the full technical plan
