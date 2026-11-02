import pytest
import torch

from src.detectors.prompts import VALID_PROMPT_MODES, build_prompts, build_text_embedding


@pytest.mark.parametrize("mode", VALID_PROMPT_MODES)
def test_prompt_modes_unit_norm(tiny_clip, mode):
    model, tok = tiny_clip
    emb = build_text_embedding(model, tok, "bottle", "cpu", prompt_mode=mode)
    assert emb.shape == (2, model.text_projection.shape[1])
    assert torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_single_differs_from_full(tiny_clip):
    model, tok = tiny_clip
    n1, a1 = build_prompts("bottle", "single")
    n3, a3 = build_prompts("bottle", "full_ensemble")
    assert len(n1) == 1 and len(n3) > len(build_prompts("bottle", "state_only")[0])
    e1 = build_text_embedding(model, tok, "bottle", "cpu", prompt_mode="single")
    e3 = build_text_embedding(model, tok, "bottle", "cpu", prompt_mode="full_ensemble")
    assert not torch.allclose(e1, e3)


def test_invalid_prompt_mode():
    with pytest.raises(ValueError):
        build_prompts("bottle", "nope")


def test_cache_key_includes_prompt_mode(tiny_clip, tmp_path):
    model, tok = tiny_clip
    build_text_embedding(model, tok, "bottle", "cpu", "single", tmp_path, "tag")
    assert (tmp_path / "text_emb_bottle_tag_single.pt").exists()


def test_set_reference_changes_map(make_winclip, images):
    det = make_winclip()
    with torch.no_grad():
        _, zs_map = det(images)
        det.set_reference(images[:1].repeat(2, 1, 1, 1) * 0.5)  # k=2
        assert det.is_fewshot and det.ref_window_features[2].shape[0] == 2 * 49
        _, fs_map = det(images)
        det.set_reference(None)
        assert not det.is_fewshot
        _, back_map = det(images)
    assert not torch.allclose(zs_map, fs_map)
    assert torch.allclose(zs_map, back_map)


@pytest.mark.parametrize("mode", ["combined", "combined_smooth"])
def test_fewshot_gradient_flow(make_winclip, images, mode):
    det = make_winclip(score_mode=mode)
    det.set_reference(images[1:])
    x = images[:1].clone().requires_grad_(True)
    score, amap = det(x)
    assert amap.shape == (1, 240, 240)
    score.sum().backward()
    assert x.grad is not None and x.grad.isfinite().all() and x.grad.abs().sum() > 0
