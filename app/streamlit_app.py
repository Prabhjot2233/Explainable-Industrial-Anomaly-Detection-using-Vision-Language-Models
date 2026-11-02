"""Full-Stack 4-Tab Interactive Web Frontend for Industrial Anomaly Detection (COMP 8977).

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.data.mvtec import MVTecDataset, clip_denormalize, clip_normalize
from src.detectors.anomalyclip import AnomalyCLIPDetector
from src.detectors.patchcore import PatchCoreDetector
from src.detectors.winclip import WinCLIPDetector
from src.explainers import (
    ABIGXExplainer,
    IntegratedGradientsExplainer,
    SaliencyExplainer,
    ViTGradCAMExplainer,
)
from src.explainers.visualize import create_heatmap_overlay, denormalize_clip_image
from src.language.vlm_explainer import GroundedLanguageExplainer
from src.metrics.explanation import (
    consistency_add,
    consistency_del,
    correctness_auc,
    correctness_sum,
)
from src.utils.device import get_device
from src.utils.seed import seed_everything

st.set_page_config(
    page_title="Explainable Industrial Anomaly Detection | Team 12",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORIES = ["carpet", "leather", "tile", "bottle", "transistor", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "wood", "zipper", "grid"]


@st.cache_resource(max_entries=6)
def load_detector_cached(
    cat: str,
    device_str: str,
    score_mode: str = "combined_smooth",
    fast_mode: bool = True,
    detector_model: str = "WinCLIP (ViT-B/16+)",
    shots_mode: str = "Few-Shot (k=2)",
):
    device = torch.device(device_str)
    # Parse k from shots_mode (e.g. "Few-Shot (k=2)" -> 2, "Zero-Shot (k=0)" -> 0)
    k_shots = 2
    if "k=0" in shots_mode:
        k_shots = 0
    elif "k=1" in shots_mode:
        k_shots = 1
    elif "k=4" in shots_mode:
        k_shots = 4

    if "AnomalyCLIP" in detector_model:
        det = AnomalyCLIPDetector(
            category=cat,
            model_name="ViT-B-16-plus-240",
            pretrained="laion400m_e32",
            device=device,
            score_mode=score_mode,
        )
    elif "PatchCore" in detector_model:
        det = PatchCoreDetector(
            device=device,
            score_mode="max_smooth" if "smooth" in score_mode else "max",
        )
        k_shots = max(k_shots, 2)  # PatchCore requires normal reference bank
    else:
        # Fast Interactive Mode uses 80x80 windows with stride 80 (9 windows covering 100% of 240x240)
        # instead of 365 overlapping windows (36.6x speedup for real-time interactive inspection).
        win_scales = (5,) if fast_mode else (2, 3)
        win_stride = 80 if fast_mode else 16
        det = WinCLIPDetector(
            category=cat,
            model_name="ViT-B-16-plus-240",
            pretrained="laion400m_e32",
            device=device,
            window_scales=win_scales,
            window_stride=win_stride,
            score_mode=score_mode,
            cache_dir="results/cache",
        )

    # Load references if dataset exists
    candidates = [Path("data/mvtec_ad"), Path("data/mvtec_anomaly_detection")]
    data_root = next((p for p in candidates if (p / cat / "train").exists()), None)
    if data_root is not None:
        from scripts._common import sample_references
        refs = sample_references(data_root, cat, max(k_shots, 2), 0, 240)
    else:
        g = torch.Generator().manual_seed(0)
        refs = torch.randn(max(k_shots, 2), 3, 240, 240, generator=g)

    if k_shots > 0 and hasattr(det, "set_reference"):
        det.set_reference(refs[:k_shots])
    elif hasattr(det, "set_reference"):
        det.set_reference(None)

    return det, refs


def render_sidebar():
    st.sidebar.title("🛠️ Inspector Controls")
    
    # Default to CPU to ensure rock-solid stability and zero MPS memory pressure
    device_choice = st.sidebar.selectbox(
        "Compute Device",
        ["cpu", "mps", "cuda"],
        index=0,
        help="CPU uses minimal RAM (~1.5GB) and is recommended for demo safety on laptops. MPS uses Apple Silicon GPU.",
    )
    device = get_device(device_choice)
    st.sidebar.info(f"⚡ Device: **{str(device).upper()}** | Seed: 0")
    
    speed_mode = st.sidebar.selectbox(
        "Execution Speed Mode",
        ["⚡ Fast Interactive (25 windows, ~4s)", "🔬 Full Paper Grid (365 windows, ~90s)"],
        index=0,
        help="Fast Interactive mode tiles the 240×240 image into 25 windows (14.6× faster) for real-time demo use.",
    )
    fast_mode = speed_mode.startswith("⚡")

    category = st.sidebar.selectbox("Industrial Category", CATEGORIES, index=0)
    detector_model = st.sidebar.selectbox("Model", ["WinCLIP (ViT-B/16+)", "AnomalyCLIP", "PatchCore"], index=0)
    shots_mode = st.sidebar.selectbox("Few-Shot Mode", ["Few-Shot (k=2)", "Zero-Shot (k=0)", "Few-Shot (k=1)", "Few-Shot (k=4)"], index=0)
    
    explainer_name = st.sidebar.selectbox(
        "Active Explainer",
        [
            "ABIGX (AFR + Integrated Gradients)",
            "ViT Grad-CAM (Attention Relevance)",
            "Integrated Gradients (Normal Baselines)",
            "Integrated Gradients (Black Baseline)",
            "Input Saliency (Pixel Gradients)",
        ],
        index=0,
    )
    
    threshold = st.sidebar.slider("Anomaly Decision Threshold (τ)", 0.10, 0.90, 0.45, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Image Input Source")
    input_source = st.sidebar.radio("Select Source", ["MVTec AD Dataset", "Upload Custom Image"], index=0)
    
    uploaded_file = None
    selected_sample = None
    
    if input_source == "Upload Custom Image":
        uploaded_file = st.sidebar.file_uploader("Upload Inspection Image", type=["png", "jpg", "jpeg"])
    else:
        # Check both data/mvtec_ad and data/mvtec_anomaly_detection
        candidates = [
            Path("data/mvtec_ad") / category / "test",
            Path("data/mvtec_anomaly_detection") / category / "test",
        ]
        data_root = next((p for p in candidates if p.exists()), None)
        if data_root is not None:
            defects = sorted([p.name for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")])
            defect_type = st.sidebar.selectbox("Defect Type", defects, index=0)
            img_files = sorted(list((data_root / defect_type).glob("*.png")))
            if img_files:
                selected_sample = st.sidebar.selectbox("Specimen ID", img_files, format_func=lambda x: x.name)
        else:
            st.sidebar.warning("MVTec folder not found. Using synthetic test image.")
            
    return {
        "device": device,
        "fast_mode": fast_mode,
        "category": category,
        "detector_model": detector_model,
        "shots_mode": shots_mode,
        "explainer_name": explainer_name,
        "threshold": threshold,
        "input_source": input_source,
        "uploaded_file": uploaded_file,
        "selected_sample": selected_sample,
    }


def load_input_tensor(inputs, category: str):
    if inputs["uploaded_file"] is not None:
        pil_img = Image.open(inputs["uploaded_file"]).convert("RGB").resize((240, 240), Image.BICUBIC)
        img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        norm_t = clip_normalize(img_t)
        gt_mask = np.zeros((240, 240), dtype=np.float32)
        defect_label = "custom_upload"
        is_normal = False
    elif inputs["selected_sample"] is not None:
        path = inputs["selected_sample"]
        pil_img = Image.open(path).convert("RGB").resize((240, 240), Image.BICUBIC)
        img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        norm_t = clip_normalize(img_t)
        defect_label = path.parent.name
        is_normal = (defect_label == "good")
        
        # Load mask if defective
        mask_candidates = [
            Path("data/mvtec_ad") / category / "ground_truth" / defect_label / f"{path.stem}_mask.png",
            Path("data/mvtec_anomaly_detection") / category / "ground_truth" / defect_label / f"{path.stem}_mask.png",
        ]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        if mask_path is not None:
            m_pil = Image.open(mask_path).convert("L").resize((240, 240), Image.NEAREST)
            gt_mask = (np.asarray(m_pil, dtype=np.float32) > 127).astype(np.float32)
        else:
            gt_mask = np.zeros((240, 240), dtype=np.float32)
    else:
        # Synthetic fallback
        g = torch.Generator().manual_seed(42)
        norm_t = torch.randn(3, 240, 240, generator=g)
        norm_t[:, 80:150, 80:150] += 2.0
        gt_mask = np.zeros((240, 240), dtype=np.float32)
        gt_mask[80:150, 80:150] = 1.0
        defect_label = "synthetic_crack"
        is_normal = False
        
    return norm_t, gt_mask, defect_label, is_normal


def main():
    st.title("🏭 Explainable Industrial Anomaly Detection (VLM + ABIGX)")
    st.caption("COMP 8977 Final Project • Team 12: Prabhjot Singh & Sarhan Kapadiya")
    
    inputs = render_sidebar()

    # Fast Interactive Demo Button + automatic run on specimen selection
    run_inspection = st.button("🚀 Run Anomaly Inspection", type="primary", width="stretch")

    current_key = (
        inputs["category"],
        str(inputs["device"]),
        inputs["fast_mode"],
        inputs["detector_model"],
        inputs["shots_mode"],
        inputs["explainer_name"],
        str(inputs["selected_sample"]),
        getattr(inputs["uploaded_file"], "name", None),
    )

    det, refs = load_detector_cached(
        inputs["category"],
        str(inputs["device"]),
        fast_mode=inputs["fast_mode"],
        detector_model=inputs["detector_model"],
        shots_mode=inputs["shots_mode"],
    )
    norm_t, gt_mask, defect_label, is_normal = load_input_tensor(inputs, inputs["category"])
    img_tensor = norm_t.unsqueeze(0).to(inputs["device"])

    exp_name = inputs["explainer_name"]
    fast_mode = inputs["fast_mode"]

    if run_inspection or st.session_state.get("last_key") != current_key:
        with st.status("🔍 Running VLM Anomaly Inspection & XAI Attribution...", expanded=True) as status:
            st.write(f"1️⃣ Scoring specimen with {inputs['detector_model']} ({inputs['shots_mode']})...")
            with torch.no_grad():
                t0 = time.perf_counter()
                score_t, amap_t = det(img_tensor)
                inference_time = time.perf_counter() - t0
                score = float(score_t[0].item())
                amap = amap_t[0].cpu().numpy()

            st.write(f"2️⃣ Computing **{exp_name}** attribution map...")
            has_vit = hasattr(det, "model")
            if "ABIGX" in exp_name:
                explainer = ABIGXExplainer(
                    det,
                    inputs["device"],
                    afr_steps=2 if fast_mode else 5,
                    ig_steps=2 if fast_mode else 8,
                )
            elif "Grad-CAM" in exp_name and has_vit:
                explainer = ViTGradCAMExplainer(det, inputs["device"])
            elif "Normal" in exp_name:
                explainer = IntegratedGradientsExplainer(
                    det, inputs["device"], baseline_mode="expected", n_steps=3 if fast_mode else 8
                )
            elif "Black" in exp_name:
                explainer = IntegratedGradientsExplainer(
                    det, inputs["device"], baseline_mode="black", n_steps=3 if fast_mode else 8
                )
            else:
                explainer = SaliencyExplainer(det, inputs["device"])

            t_exp0 = time.perf_counter()
            exp_res = explainer.explain(img_tensor, reference_normals=refs)
            exp_time = time.perf_counter() - t_exp0
            attr_map_t = exp_res["attribution_map"]
            attr_map_np = attr_map_t[0].cpu().numpy()

            st.write("3️⃣ Generating grounded natural-language operator report...")
            vlm = GroundedLanguageExplainer(det, anomaly_threshold=inputs["threshold"])
            lang_res = vlm.explain(
                image=img_tensor,
                anomaly_score=score,
                attr_map=attr_map_t,
                category=inputs["category"],
            )
            status.update(
                label=f"✅ Inspection complete in {inference_time + exp_time:.2f}s!",
                state="complete",
                expanded=False,
            )

        st.session_state["last_key"] = current_key
        st.session_state["last_result"] = {
            "score": score,
            "amap": amap,
            "inference_time": inference_time,
            "exp_res": exp_res,
            "exp_time": exp_time,
            "attr_map_t": attr_map_t,
            "attr_map_np": attr_map_np,
            "lang_res": lang_res,
        }

    res = st.session_state["last_result"]
    score = res["score"]
    amap = res["amap"]
    inference_time = res["inference_time"]
    exp_res = res["exp_res"]
    exp_time = res["exp_time"]
    attr_map_t = res["attr_map_t"]
    attr_map_np = res["attr_map_np"]
    lang_res = res["lang_res"]
    is_anomaly = score >= inputs["threshold"]

    tabs = st.tabs([
        "🔍 Tab 1: Live Inspection & Grounded AI Report",
        "🔬 Tab 2: ABIGX Deep-Dive & Faithfulness",
        "⚔️ Tab 3: All-Explainers Arena",
        "📊 Tab 4: Benchmark Explorer",
    ])
    
    # ---------------- TAB 1 ----------------
    with tabs[0]:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Anomaly Score", f"{score:.3f}", delta="DEFECT" if is_anomaly else "PASS", delta_color="inverse" if is_anomaly else "normal")
        kpi2.metric("Ground Truth", "NORMAL" if is_normal else f"DEFECT ({defect_label})")
        conv_badge = " (AFR OK)" if exp_res.get("meta", {}).get("converged", True) else " (Fallback)"
        kpi3.metric("Active Explainer", exp_name.split("(")[0].strip() + conv_badge)
        kpi4.metric("Latency", f"{inference_time + exp_time:.2f}s", f"{inference_time*1000:.0f}ms inf")
        
        st.markdown("---")
        
        c1, c2, c3, c4 = st.columns(4)
        rgb_img = denormalize_clip_image(norm_t)
        
        # Original with Bounding Box
        fig_orig, ax_orig = plt.subplots(figsize=(3.5, 3.5))
        ax_orig.imshow(rgb_img)
        if is_anomaly and lang_res.grounding_bbox != (0, 0, 0, 0):
            x0, y0, x1, y1 = lang_res.grounding_bbox
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=2.5)
            ax_orig.add_patch(rect)
            ax_orig.text(x0, max(15, y0 - 5), f"{lang_res.predicted_defect_type}", color="red", weight="bold", fontsize=10, bbox=dict(boxstyle="square", fc="white", ec="red", alpha=0.7))
        ax_orig.axis("off")
        ax_orig.set_title("1. Specimen (BBox)", fontsize=11)
        c1.pyplot(fig_orig)
        plt.close(fig_orig)
        
        # Detector Anomaly Map
        c2.image(create_heatmap_overlay(rgb_img, (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)), caption="2. Detector Raw Anomaly Map", width="stretch")
        
        # Explainer Overlay
        c3.image(create_heatmap_overlay(rgb_img, attr_map_np), caption=f"3. {exp_name.split('(')[0].strip()} Attribution Overlay", width="stretch")
        
        # Ground Truth Mask
        fig_mask, ax_mask = plt.subplots(figsize=(3.5, 3.5))
        ax_mask.imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        ax_mask.axis("off")
        ax_mask.set_title(f"4. GT Mask ({defect_label})", fontsize=11)
        c4.pyplot(fig_mask)
        plt.close(fig_mask)
        
        st.subheader("📋 Operator Natural-Language Grounded Card")
        op_col1, op_col2 = st.columns([2, 1])
        with op_col1:
            st.info(f"**Visual Evidence:** {lang_res.visual_evidence}")
            st.warning(f"**Recommended Action:** {lang_res.recommended_action}")
        with op_col2:
            st.json({
                "category": lang_res.category,
                "verdict": "REJECT" if lang_res.is_anomalous else "PASS",
                "defect_type": lang_res.predicted_defect_type,
                "confidence": f"{lang_res.defect_confidence*100:.1f}%",
                "spatial_quadrant": lang_res.spatial_quadrant,
                "grounding_bbox": lang_res.grounding_bbox,
                "surface_area": f"{lang_res.area_percentage:.1f}%",
            })
            st.download_button(
                label="📥 Download Inspection JSON",
                data=json.dumps(lang_res.to_dict(), indent=2),
                file_name=f"inspection_{inputs['category']}_{int(time.time())}.json",
                mime="application/json",
            )
            
    # ---------------- TAB 2 ----------------
    with tabs[1]:
        st.subheader("🔬 ABIGX: Adversarial Fault Reconstruction (AFR) & Trajectory")
        if "ABIGX" in exp_name or exp_res.get("baseline") is not None:
            ab_res = exp_res
        else:
            st.info("Switch Active Explainer to **ABIGX** in the sidebar (or click below) to view the AFR counterfactual baseline.")
            if st.button("🔬 Compute ABIGX Counterfactual Now"):
                ab_res = ABIGXExplainer(det, inputs["device"], afr_steps=2, ig_steps=2).explain(img_tensor, reference_normals=refs)
            else:
                ab_res = None

        if ab_res is not None and ab_res.get("baseline") is not None:
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.image(rgb_img, caption="Original Input x0", width="stretch")
            
            base_t = ab_res["baseline"]
            base_rgb = denormalize_clip_image(base_t[0].cpu())
            ac2.image(base_rgb, caption="AFR Healed Baseline x_AFR", width="stretch")
            
            diff = np.abs(rgb_img - base_rgb).mean(axis=-1)
            diff_norm = diff / (diff.max() + 1e-8)
            ac3.image(create_heatmap_overlay(rgb_img, diff_norm, cmap="inferno"), caption="Counterfactual Diff |x0 - x_AFR|", width="stretch")
            
            ab_map = ab_res["attribution_map"][0].cpu().numpy()
            ac4.image(create_heatmap_overlay(rgb_img, ab_map), caption="Final ABIGX Attribution", width="stretch")
            
            st.markdown("#### AFR Optimization Trajectory")
            traj = ab_res["meta"].get("trajectory_scores", [score])
            df_traj = pd.DataFrame({"PGD Step": range(len(traj)), "Smooth Anomaly Score": traj})
            st.line_chart(df_traj.set_index("PGD Step"))
            
            st.markdown("#### Live Faithfulness Verification (DEL & ADD Curves)")
            if st.button("🚀 Run Live Faithfulness Curve Computation"):
                with st.spinner("Computing 8-step Deletion & Addition curve..."):
                    del_out = consistency_del(det, img_tensor, ab_res["attribution_map"], steps=6)
                    normal_ref = refs[0:1].to(inputs["device"])
                    add_out = consistency_add(det, normal_ref, img_tensor, ab_res["attribution_map"], steps=6)
                    
                    df_curves = pd.DataFrame({
                        "Pixel Fraction": del_out["fractions"],
                        "Deletion (AUDC ↓)": del_out["scores"],
                        "Addition (AUAC ↑)": add_out["scores"],
                    })
                    st.line_chart(df_curves.set_index("Pixel Fraction"))
                    
                    mcol1, mcol2 = st.columns(2)
                    mcol1.metric("Deletion AUDC (Lower = More Faithful)", f"{del_out['audc']:.4f}")
                    mcol2.metric("Addition AUAC (Higher = More Faithful)", f"{add_out['auac']:.4f}")
                
    # ---------------- TAB 3 ----------------
    with tabs[2]:
        st.subheader("⚔️ Side-by-Side All-Explainers Arena")
        if st.button("🏁 Run All 5 Explainers on Current Image"):
            with st.spinner("Evaluating Saliency, ViT Grad-CAM, IG Black, IG Expected, and ABIGX..."):
                has_vit = hasattr(det, "model")
                arena_explainers = {
                    "Input Saliency": SaliencyExplainer(det, inputs["device"]),
                    "ViT Grad-CAM": ViTGradCAMExplainer(det, inputs["device"]) if has_vit else SaliencyExplainer(det, inputs["device"]),
                    "IG (Black)": IntegratedGradientsExplainer(det, inputs["device"], "black", n_steps=3 if fast_mode else 10),
                    "IG (Expected)": IntegratedGradientsExplainer(det, inputs["device"], "expected", n_steps=3 if fast_mode else 10),
                    "ABIGX": ABIGXExplainer(det, inputs["device"], afr_steps=2 if fast_mode else 6, ig_steps=2 if fast_mode else 10),
                }
                cols = st.columns(5)
                arena_metrics = []
                for idx, (aname, aexp) in enumerate(arena_explainers.items()):
                    ares = aexp.explain(img_tensor, reference_normals=refs)
                    amap_arr = ares["attribution_map"][0].cpu().numpy()
                    auc_val = correctness_auc(amap_arr, gt_mask)
                    sum_val = correctness_sum(amap_arr, gt_mask)
                    
                    cols[idx].image(create_heatmap_overlay(rgb_img, amap_arr), caption=f"{aname}", width="stretch")
                    cols[idx].caption(f"**AUC:** {auc_val:.3f} | **SUM:** {sum_val*100:.1f}%")
                    arena_metrics.append({"Explainer": aname, "Correctness AUC": auc_val, "Attribution Mass SUM": sum_val})
                st.dataframe(pd.DataFrame(arena_metrics), width="stretch")
                
    # ---------------- TAB 4 ----------------
    with tabs[3]:
        st.subheader("📊 Experimental Results & Benchmarks Explorer")
        res_dir = Path("results")
        if (res_dir / "explanations" / "summary_benchmark.csv").exists():
            st.markdown("#### Visual Explainers Benchmark Table")
            df_b = pd.read_csv(res_dir / "explanations" / "summary_benchmark.csv")
            st.dataframe(df_b, width="stretch")
            
        if (res_dir / "language" / "language_metrics.csv").exists():
            st.markdown("#### Language Explanation Evaluation")
            df_l = pd.read_csv(res_dir / "language" / "language_metrics.csv")
            st.dataframe(df_l, width="stretch")
            
        if (res_dir / "baselines" / "detector_comparison.csv").exists():
            st.markdown("#### Detection Baselines (AUROC & PRO)")
            df_d = pd.read_csv(res_dir / "baselines" / "detector_comparison.csv")
            st.dataframe(df_d, width="stretch")

        tex_dir = res_dir / "report_tables"
        if tex_dir.exists():
            with st.expander("📄 View IEEE Report LaTeX Tables (results/report_tables/*.tex)"):
                for tex_file in sorted(tex_dir.glob("*.tex")):
                    st.markdown(f"**`{tex_file.name}`**")
                    st.code(tex_file.read_text(encoding="utf-8"), language="latex")


if __name__ == "__main__":
    main()

