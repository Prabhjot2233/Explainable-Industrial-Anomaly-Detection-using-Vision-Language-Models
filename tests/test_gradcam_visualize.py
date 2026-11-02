import torch

from src.explainers.gradcam_vit import ViTGradCAMExplainer
from src.explainers.visualize import save_explanation_comparison_grid


# ---------------- Part A: ViT Grad-CAM
def test_gradcam_shape_range_and_hooks_removed(make_winclip, images):
    det = make_winclip()
    block = det.model.visual.transformer.resblocks[-1]
    out = ViTGradCAMExplainer(det, torch.device("cpu")).explain(images[:1])
    m = out["attribution_map"]
    assert m.shape == (1, 240, 240)
    assert m.isfinite().all() and m.min() >= 0 and m.max() <= 1
    assert out["meta"]["grid"] == 15
    assert len(block._forward_hooks) == 0
    assert len(block._backward_hooks) == 0


# ---------------- Part B: visualization
def test_save_comparison_grid(tmp_path):
    g = torch.Generator().manual_seed(0)
    img = torch.randn(3, 240, 240, generator=g)
    mask = torch.zeros(1, 240, 240)
    mask[:, 80:140, 80:140] = 1
    amap = torch.rand(240, 240, generator=g)
    attrs = {n: torch.rand(1, 240, 240, generator=g)
             for n in ["Saliency", "ViT Grad-CAM", "IG (Black)", "Expected-IG (Normal)"]}
    path = save_explanation_comparison_grid(img, mask, amap, attrs, tmp_path / "grid.png", "test")
    assert path.exists() and path.stat().st_size > 10 * 1024
