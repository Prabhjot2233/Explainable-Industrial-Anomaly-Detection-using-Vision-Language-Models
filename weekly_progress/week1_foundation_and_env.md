# Week 1: Foundation, Environment and Metrics (Steps 1–3)

| Step | Status |
|---|---|
| 1. Utilities: device, seed, config | ✅ Done |
| 2. Environment and gradient check | ⏳ Next |
| 3. Detection metrics (AUROC, PRO) | ⬜ To do |

---

## Step 1: Utilities, Configuration and Tracking Setup ✅

### What we built and files created
| File | Purpose |
|---|---|
| `src/utils/device.py` | `get_device()` picks where to run: NVIDIA GPU, Apple GPU, or CPU |
| `src/utils/seed.py` | `seed_everything()` makes random numbers repeatable |
| `src/utils/config.py` | `ZeroShotConfig` plus `load_config()` / `save_config()` for YAML settings |
| `configs/zeroshot.yaml` | The default settings for every experiment |
| `tests/test_utils_config.py` | Automatic checks that the three utilities work |
| `weekly_progress/*` | These beginner guides |

### Why we needed it (the big picture)
Every later step needs to know **where to run** (GPU or CPU), must give **the same answer
every time** (so results are trustworthy), and needs **settings** like image size or model
name. Putting these in one shared place means no step reinvents them, and the whole team
uses identical settings.

### How it works under the hood
- **Device:** `get_device()` asks PyTorch "is there an NVIDIA GPU?" (`torch.cuda.is_available()`),
  then "is there an Apple GPU?" (`torch.backends.mps.is_available()`), and otherwise uses
  the CPU. You can force one with `get_device("cpu")`. If you ask for a GPU you don't have,
  it quietly falls back to the CPU instead of crashing.
- **Seed:** randomness in Python, NumPy and PyTorch each has its own "dice". `seed_everything(0)`
  sets all of them to the same starting point, so `torch.rand(3, 3)` gives identical numbers
  every run.
- **Config:** `ZeroShotConfig` is a Python *dataclass*, a labelled box of settings with
  defaults. `save_config` writes it to a YAML text file and `load_config` reads it back.
  Typos are caught early: unknown keys or an invalid `score_mode` raise an error immediately.

Key default settings (`configs/zeroshot.yaml`):

| Setting | Value | Meaning |
|---|---|---|
| `model` | `ViT-B-16-plus-240` | Which CLIP architecture |
| `pretrained` | `laion400m_e32` | Which trained weights |
| `image_size` | `240` | Every image is resized to 240×240 |
| `window_scales` | `[2, 3]` | Window sizes of 2×2 and 3×3 patches (32px and 48px) |
| `window_chunk_size` | `16` | How many windows go through CLIP at once (limits memory) |
| `num_workers` | `0` | Load data in the main process (avoids Windows/Mac hangs) |
| `seed` | `0` | Fixed randomness |
| `score_mode` | `combined` | How the final score is computed (`global`, `map_max`, `combined`, `combined_smooth`) |

### Key engineering decisions
- **Mac:** when MPS is chosen we set `PYTORCH_ENABLE_MPS_FALLBACK=1`, so any operation Apple's
  GPU doesn't support runs on the CPU instead of crashing.
- **Windows:** files are always opened with `encoding="utf-8"` (Windows otherwise defaults to
  `cp1252` and can crash), paths use `pathlib.Path`, and `num_workers=0` avoids
  multiprocessing hangs.
- **Week 4 preparation:** everything stays `float32` for stable gradients.

### How to run and verify
```bash
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pytest tests/test_utils_config.py -v
```
Expected output:
```
tests/test_utils_config.py::test_device_selection PASSED
tests/test_utils_config.py::test_seed_reproducibility PASSED
tests/test_utils_config.py::test_config_io PASSED
tests/test_utils_config.py::test_default_yaml_matches_plan PASSED
tests/test_utils_config.py::test_invalid_score_mode_rejected PASSED
5 passed
```

---

## Step 2: Environment and Differentiability Check ⏳

**Task:** create `scripts/check_env.py`, which:
1. Prints Python, OS, PyTorch versions and available devices.
2. Lists `open_clip` pretrained tags for `ViT-B-16-plus-240` (confirms `laion400m_e31` and `e32` exist).
3. Loads the model frozen, in `float32` and `.eval()` mode.
4. Feeds a random image with `requires_grad=True`, runs `encode_image`, and calls `.backward()`.
5. Checks that the image gradient exists, is finite and is non-zero. Exits with 0 on pass, 1 on fail.

**Why:** proves on day 1 that this machine can run CLIP *and* pass gradients back to pixels,
which Week 4 explanations depend on.

*(Full beginner explanation is added when the step is complete.)*

---

## Step 3: Detection Metrics ⬜

**Task:** `src/metrics/detection.py` with `image_auroc`, `pixel_auroc`, `pro_score` and
`evaluate_category`, plus `tests/test_metrics.py` using synthetic data (random, perfect,
inverted, hand-calculated toy example, edge cases).

**Why:** we need a trustworthy scoreboard before building the detector.

*(Full beginner explanation and AUROC vs PRO diagram are added when the step is complete.)*
