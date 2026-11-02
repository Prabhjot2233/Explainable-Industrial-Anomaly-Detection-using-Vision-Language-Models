import sys
from pathlib import Path

# Make the repo root importable so `import src...` works under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
import torch  # noqa: E402

# Tiny random-weight CLIP with the same geometry as ViT-B-16-plus-240 (240px, patch 16,
# 15x15 tokens) so tests run in seconds without downloading pretrained weights.
TINY_CLIP = "TinyCLIP-16-240-test"
TINY_CLIP_CFG = {
    "embed_dim": 64,
    "vision_cfg": {"image_size": 240, "layers": 2, "width": 64, "head_width": 32, "patch_size": 16},
    "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 64, "heads": 2, "layers": 2},
}


def _register_tiny_clip() -> None:
    import open_clip.factory as factory

    factory._MODEL_CONFIGS[TINY_CLIP] = TINY_CLIP_CFG


@pytest.fixture(scope="session")
def tiny_clip():
    from src.detectors.winclip import load_clip

    _register_tiny_clip()
    torch.manual_seed(0)
    return load_clip(TINY_CLIP, None, torch.device("cpu"))


@pytest.fixture
def make_winclip(tiny_clip):
    from src.detectors.winclip import WinCLIPDetector

    model, tok = tiny_clip

    def _make(**kw):
        args = dict(category="bottle", pretrained=None, device="cpu", window_scales=[2],
                    window_stride=32, window_chunk_size=64, cache_dir=None,
                    model=model, tokenizer=tok)
        args.update(kw)
        return WinCLIPDetector(**args)

    return _make


@pytest.fixture
def images():
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 3, 240, 240, generator=g)
