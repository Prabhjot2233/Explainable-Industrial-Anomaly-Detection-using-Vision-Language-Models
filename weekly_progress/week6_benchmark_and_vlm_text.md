# Week 6: Explanation Benchmark & Grounded Language Explanations (Steps 20–21)

| Step | Task | Key files | Status |
|---|---|---|---|
| 20 | Explanation Benchmark & AFR Ablation Runner | `configs/explanations.yaml`, `scripts/run_explanations.py` | ✅ Done |
| 21 | Grounded VLM & OpenCLIP Language Explainer | `src/language/vlm_explainer.py`, `configs/language.yaml`, `tests/test_language_explainer.py` | ✅ Done |

---

## Step 20: Explanation Benchmark & AFR Ablation

### 1. Plain-English Analogy
In a production quality audit, having one explanation method isn't enough; we need to compare all available tools side-by-side. Step 20 evaluates all 5 visual explanation methods:
- **Input Saliency** (Pixel-level sensitivity)
- **ViT Grad-CAM** (Transformer attention relevance)
- **Integrated Gradients (Black Baseline)** (Linear path from zeros)
- **Expected-IG** (Average path from real normal images)
- **ABIGX** (Path from counterfactual AFR healed baseline)

We compute their **Correctness AUC**, **Correctness SUM**, **Deletion AUDC**, and **Addition AUAC** across test samples. We also run an ablation study varying AFR steps and learning rate to discover optimal convergence settings.

### 2. Output Artifacts
- `results/explanations/summary_benchmark.csv`
- `results/explanations/<category>/metrics_per_image.csv`
- `results/explanations/afr_ablation.csv`
- Multi-panel comparison grids under `results/explanations/<category>/panels/`

---

## Step 21: Grounded Language Explanations

### 1. Plain-English Analogy
A heatmap showing red and yellow glowing spots tells an engineer *where* something is unusual, but an operator on the factory floor needs words:
> *"Cracked bottle neck in the top-left quadrant covering 3.2% of the surface. Immediate quarantine required."*

Step 21 translates mathematical attributions into actionable natural-language reports with zero hallucination.

### 2. How the Engineering Works
1. **Hallucination Guardrail:** If the model's anomaly score is below the decision threshold ($\tau = 0.45$), it refuses to invent a defect and returns a clean `PASS` verdict.
2. **Spatial Grounding:** Connected component analysis on the attribution map finds the bounding box, area share, and quadrant (`top-left`, `center`, etc.).
3. **Crop-Level Defect Classification:** The padded bounding box crop is classified against canonical defect text prompts (`cut`, `hole`, `scratch`, `contamination`, `crack`, `deformation`) using OpenCLIP embeddings.
4. **Actionable Recommendations:** Outputs specific engineering guidance for each defect family.

### 3. Verification Command
```bash
python -m pytest tests/test_language_explainer.py -v
```

