# Week 7: Language Evaluation & Frontend Playbook (Steps 22–23)

| Step | Task | Key files | Status |
|---|---|---|---|
| 22 | Language Explanation Evaluation Suite | `src/metrics/language_eval.py`, `scripts/run_language_eval.py`, `tests/test_language_metrics.py` | ✅ Done |
| 23 | Full-Stack Interactive 4-Tab Streamlit Frontend | `app/streamlit_app.py`, `tests/test_frontend_smoke.py` | ⏳ In Progress |

---

## Step 22: Language Evaluation Suite (Accuracy, Grounding & Hallucinations)

### 1. Plain-English Analogy
In mission-critical quality control, a natural-language AI explanation cannot afford to make things up (*hallucinate*). If an AI claims a bottle is broken when it's flawless, or claims a crack is on the bottom-right when it's on the neck, operators will lose trust.

Step 22 quantitatively measures:
- **Defect Family Accuracy:** Did the model correctly categorize the defect family (e.g. `crack`, `hole`, `cut`, `scratch`)?
- **Quadrant Grounding Agreement:** Does the spatial quadrant reported by the AI match the human ground-truth defect mask?
- **Bounding-Box IoU:** How accurately does the AI's bounding box envelop the defect?
- **Hallucination Rates:**
  - False positive defect claims on clean normal images.
  - Spatial hallucination (claiming a defect location that has 0% overlap with the actual ground-truth mask).

### 2. Output Artifacts
- `results/language/language_metrics.csv`
- `results/language/predictions.jsonl`
- `results/language/human_rating_rubric.csv` (20-sample blinded sheet for qualitative human rating)

### 3. Verification Command
```bash
python -m pytest tests/test_language_metrics.py -v
```

