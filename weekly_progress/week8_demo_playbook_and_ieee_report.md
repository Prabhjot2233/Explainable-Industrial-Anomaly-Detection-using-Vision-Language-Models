# Week 8: Final Demo Playbook & IEEE Report Artifacts

**Team 12:** Prabhjot Singh & Sarhan Kapadiya  
**Course:** COMP 8977 — Project Phase II  
**Supervisors:** Dr. Shaon Shuvo & Dr. Prashanth Cheluvasai Ranga  

---

## 1. 5-Minute Live Demo Script (Step-by-Step for Prabhjot & Sarhan)

### Step 1: Launch the Interactive App
In your terminal, activate your virtual environment and run:
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

### Step 2: Tab 1 — Live Inspection & Grounded AI Report (2 minutes)
* **What to Show:**
  1. Select a category (e.g. `bottle` or `carpet`) from the sidebar.
  2. Choose a defect sample from the dropdown (e.g., `broken_large` or `hole`).
  3. Point out the **Top KPI Banner**:
     - Live Anomaly Score (e.g., `0.912` $\rightarrow$ `DEFECT DETECTED`).
     - Real-time inference latency (measured on Apple MPS / CUDA).
  4. Walk through the **4-Column Visual Inspection Grid**:
     - *Col 1 (Input + BBox):* Show the original specimen with the red grounded bounding box drawn dynamically around the defect.
     - *Col 2 (Detector Map):* Show WinCLIP's raw harmonic-mean anomaly heatmap.
     - *Col 3 (Explainer Heatmap):* Show the active explainer attribution (e.g. ABIGX).
     - *Col 4 (Ground Truth):* Compare directly against human ground truth to prove pixel precision.
  5. Read the **Operator Grounded Narrative Card**:
     - Point out that the model classifies the specific defect family (e.g. `"crack"` or `"hole"`), reports the spatial quadrant (`"top-left"`), the exact surface area share (`3.2%`), and gives actionable engineering advice (`Quarantine immediately`).
     - Click **"Download Inspection JSON"** to demonstrate integration into automated manufacturing MES / ERP systems.

---

### Step 3: Tab 2 — ABIGX Deep-Dive: Adversarial Fault Reconstruction (1.5 minutes)
* **What to Explain to Dr. Shuvo & Dr. Ranga:**
  > *"Standard Integrated Gradients suffers from the baseline problem: using an artificial black image measures the difference between a product and darkness, highlighting the entire object body. Expected-IG averages normal images, but normal images may have different alignments or textures.*  
  > *Our research contribution, **ABIGX**, uses Projected Gradient Descent (PGD) to minimally heal the defect until the anomaly score drops below 0.35. Integrating gradients between this counterfactual healed baseline and the defective input highlights **only the anomaly itself**."*
* **What to Click:**
  1. Show the 4-panel progression: `Input` $\rightarrow$ `AFR Healed Baseline` $\rightarrow$ `Pixel Diff |x0 - x_AFR|` $\rightarrow$ `Final ABIGX Map`.
  2. Show the **AFR PGD Optimization Curve** demonstrating the smooth score dropping monotonically across iterations.
  3. Click **"Run Live Faithfulness Check"**: Show the real-time Deletion Curve (AUDC drops rapidly) and Addition Curve (AUAC rises rapidly) proving high fidelity.

---

### Step 4: Tab 3 — Side-by-Side All-Explainers Arena (1 minute)
* Click **"Run All 5 Explainers on Current Image"**.
* Show the comparative grid:
  `Input Saliency` vs. `ViT Grad-CAM` vs. `IG (Black)` vs. `IG (Expected)` vs. `ABIGX`.
* Highlight that ABIGX achieves higher **Correctness AUC** and tighter **Attribution Mass SUM** with significantly less background noise.

---

### Step 5: Tab 4 — Benchmark Explorer (30 seconds)
* Show the comprehensive benchmark tables compiled directly from the MVTec AD runs (Detection Baselines, Visual Explainer Metrics, and Language Hallucination evaluation).

---

## 2. Top 10 Supervisor Q&A Preparation

### Q1: Why did you choose Vision-Language Models (CLIP) instead of supervised defect detectors?
**Answer:** Supervised defect models require thousands of labeled defect examples per product. In real factories, defects are rare, unpredictable, and constantly evolving. CLIP allows zero-shot and few-shot detection by leveraging internet-scale visual concepts of "damaged", "scratched", and "broken", requiring only 0 to 4 normal reference photos.

### Q2: What is the computational cost of WinCLIP's multi-scale sliding windows?
**Answer:** A standard $240 \times 240$ image generates ~365 sliding windows across scales 2 and 3. To maintain interactive speeds, we chunk window encoding and reuse the window embeddings for both the language scoring branch ($M_{\text{text}}$) and the few-shot visual reference branch ($M_{\text{vis}}$). This cuts inference time by half.

### Q3: What is the "baseline problem" in Integrated Gradients?
**Answer:** The Integrated Gradients path integral requires a reference baseline representing the absence of the feature of interest. Using an all-black baseline measures the difference between an industrial object and blackness, causing non-defective edges and background boundaries to receive high attribution.

### Q4: How does ABIGX solve the baseline problem?
**Answer:** ABIGX (Adversarial Fault Reconstruction) formulates baseline selection as an optimization problem: we perturb the anomalous image within an $L_\infty$ ball using PGD to minimize the anomaly score. The resulting counterfactual image preserves identical product geometry and lighting while eliminating the defect.

### Q5: What happens if AFR fails to converge or find a healed baseline?
**Answer:** We implemented the mitigation proposed in Section 4.4 of our proposal: if the relative score drop during PGD is below 15%, the explainer flags `converged = False` and automatically falls back to Expected Integrated Gradients over stored normal references.

### Q6: What do the Deletion (DEL) and Addition (ADD) metrics measure?
**Answer:** They evaluate explanation faithfulness (model fidelity) without relying on ground-truth human masks. Deletion erases the most important pixels first (a faithful explainer causes the anomaly score to plummet, yielding low AUDC). Addition inserts those pixels into a clean image (a faithful explainer causes the anomaly score to jump immediately, yielding high AUAC).

### Q7: How do you prevent your Language Explainer from hallucinating defects?
**Answer:** We enforce a strict two-stage guardrail: if the global anomaly score is below the calibrated threshold ($\tau = 0.45$), the language module is blocked from generating defect descriptions and emits an affirmative `PASS` verdict. For anomalous images, defect descriptions are conditioned directly on cropped, high-attribution regions.

### Q8: How did you evaluate the natural language explanations?
**Answer:** We designed a multi-metric protocol in `src/metrics/language_eval.py`:
1. *Defect Family Accuracy* against MVTec ground truth labels.
2. *Spatial Quadrant Match* and *Bounding-Box IoU* against ground-truth segmentation masks.
3. *Hallucination Rate* on normal images (false positive defect claims) and anomalous images (zero-IoU spatial claims).

### Q9: Can this system run entirely on an edge device or laptop without GPU server access?
**Answer:** Yes. The entire stack runs on Apple Silicon (`mps`) or CPU with no external cloud dependencies. We use an OpenCLIP-grounded visual classifier for defect naming, bypassing the need for a 16 GB VRAM LLM server.

### Q10: How does your work compare to PatchCore?
**Answer:** PatchCore is an excellent memory-bank baseline that achieves high AUROC, but it has no semantic understanding—it cannot explain in words *what* is wrong or adapt to textual defect specifications. Our vision-language approach provides both competitive detection and explainability (visual heatmaps + natural-language operator instructions).

