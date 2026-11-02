import pytest


@pytest.mark.parametrize("mode", ["global", "map_max", "combined", "combined_smooth"])
def test_winclip_output_shapes(make_winclip, images, mode):
    det = make_winclip(score_mode=mode, window_scales=[2, 3])
    score, amap = det(images)
    assert score.shape == (2,)
    assert amap.shape == (2, 240, 240)
    assert score.isfinite().all() and amap.isfinite().all()
