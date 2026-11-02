def test_winclip_input_gradients(make_winclip, images):
    det = make_winclip(score_mode="combined_smooth")
    x = images.clone().requires_grad_(True)
    score, _ = det(x)
    score.sum().backward()
    assert x.grad is not None
    assert x.grad.isfinite().all()
    assert x.grad.abs().sum() > 0
