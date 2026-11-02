# Week 5: ABIGX & Quantitative XAI Metrics (Steps 18–19)

| Step | Task | Key files | Status |
|---|---|---|---|
| 18 | ABIGX (Adversarial Fault Reconstruction + IG) | `src/explainers/abigx.py`, `tests/test_abigx.py` | ✅ Done |
| 19 | Quantitative XAI Metrics (AUC, SUM, DEL, ADD) | `src/metrics/explanation.py`, `tests/test_explanation_metrics.py` | ✅ Done |

---

## Step 18: ABIGX-Style Explainer (AFR + Integrated Gradients)

### 1. Plain-English Analogy
Standard Integrated Gradients (IG) draws a straight line between a baseline image and the defective test photo. But if you use an artificial black baseline, you are asking: *"What makes this image different from total darkness?"* This highlights the entire product body rather than just the defect!

**ABIGX** (Zhuo et al., 2026) solves this with **Adversarial Fault Reconstruction (AFR)**:
Imagine a digital photo editor with a tiny airbrush. It makes the absolute smallest modifications necessary to "heal" the crack or hole until the anomaly detector considers the image completely normal. That healed image is the **counterfactual baseline** ($x_{\text{AFR}}$). Integrating gradients between the healed image and the defective image pinpoints **only the fault itself**!

### 2. How the Math Works
1. **Stage 1 — AFR via PGD:**
   Given anomalous input $x^{(0)}$, we minimize:
   $$\mathcal{L}_{\text{AFR}}(x) = F_{\text{smooth}}(x) + \lambda_{\text{prox}} \|x - x^{(0)}\|_2^2$$
   subject to $\|x - x^{(0)}\|_\infty \le \epsilon$ and valid pixel bounds.
   - We stop as soon as $F_{\text{smooth}}(x) \le 0.35$ (normal score threshold).
2. **Mitigation / Fallback:**
   If the score drop $\Delta_{\text{rel}} < 15\%$, AFR has struggled to heal the image. We automatically fall back to **Expected-IG** over normal reference images.
3. **Stage 2 — Path Integration:**
   We integrate gradients along the trajectory $x_{\text{AFR}} \rightarrow x^{(0)}$ with $M$ steps and compute the channel L2 norm.

### 3. Verification Command
```bash
python -m pytest tests/test_abigx.py -v
```

---

## Step 19: Quantitative Explanation Metrics (Correctness & Consistency)

### 1. Plain-English Analogy
How do we know which explainer is telling the truth? We evaluate two dimensions:
1. **Correctness (Ground-Truth Localization):** Does the heatmap highlight the ground-truth mask marked by human inspectors?
2. **Consistency / Faithfulness (Model Fidelity):** If we erase the pixels highlighted by the explainer, does the model's anomaly score drop immediately? If we paste those pixels onto a normal image, does the score spike immediately?

### 2. Math Formulations
- **Correctness AUC:** ROC-AUC between attribution map and binary mask.
- **Correctness SUM:**
  $$\text{SUM} = \frac{\sum_{p \in \text{mask}} |A(p)|}{\sum_{\text{all } p} |A(p)|}$$
  Measures the fraction of attribution mass concentrated inside the defect.
- **Deletion (DEL) AUDC:** Progressively deletes top-attributed pixels. Lower Area Under the Deletion Curve = better explanation.
- **Addition (ADD) AUAC:** Progressively inserts top-attributed defect pixels into a normal image. Higher Area Under the Addition Curve = better explanation.

### 3. Verification Command
```bash
python -m pytest tests/test_explanation_metrics.py -v
```

