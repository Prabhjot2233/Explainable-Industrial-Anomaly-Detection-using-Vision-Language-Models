# Week 2: Dataset and WinCLIP Detector (Steps 4–10)

Nothing here is built yet. This is the task list; each section gets a full beginner
explanation (what / why / how / decisions / how to run) as it's completed.

| Step | Task | Key files | Status |
|---|---|---|---|
| 4 | **MVTec AD loader**: images resized to 240×240 and CLIP-normalised, binary masks | `src/data/mvtec.py`, `tests/test_dataset.py` | ⬜ |
| 5 | **Prompts and detector skeleton**: "flawless X" vs "damaged X" text embeddings | `src/detectors/base.py`, `prompts.py`, `winclip.py` | ⬜ |
| 6 | **Global scoring**: whole image vs prompts gives an anomaly probability | `src/detectors/winclip.py` | ⬜ |
| 7 | **Window scoring**: sliding 32px/48px windows, harmonic mean, heatmap | `src/detectors/winclip.py` | ⬜ |
| 8 | **Gradient flow and device tests**: gradients reach pixels; Mac/GPU matches CPU | `tests/test_grad_flow.py`, `tests/test_device_consistency.py` | ⬜ |
| 9 | **Zero-shot runner**: evaluate categories and write `metrics.csv` | `scripts/run_zeroshot.py` | ⬜ |
| 10 | **Checkpoint comparison**: `laion400m_e31` vs `e32`, compared with the paper | `scripts/compare_checkpoints.py` | ⬜ |

**Target to sanity-check against (WinCLIP paper, MVTec AD zero-shot):**
Image AUROC 91.8%, Pixel AUROC 85.1%, PRO 64.6%.

**Before Step 4:** download MVTec AD (~5 GB) from
https://www.mvtec.com/company/research/datasets/mvtec-ad and extract it into `data/mvtec_ad/`.
