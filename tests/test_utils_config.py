# tests/test_utils_config.py
import pathlib

import pytest
import torch
import yaml

from src.utils.config import ZeroShotConfig, load_config, save_config
from src.utils.device import get_device
from src.utils.seed import seed_everything

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_device_selection():
    assert get_device("cpu") == torch.device("cpu")
    expected_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device("cuda").type == expected_cuda
    assert isinstance(get_device(), torch.device)
    with pytest.raises(ValueError):
        get_device("tpu")


def test_seed_reproducibility():
    seed_everything(1234)
    a = torch.rand(3, 3)
    seed_everything(1234)
    b = torch.rand(3, 3)
    assert torch.allclose(a, b)


def test_config_io(tmp_path: pathlib.Path):
    cfg = ZeroShotConfig(image_size=256, seed=42, score_mode="global")
    file_path = tmp_path / "cfg.yaml"
    save_config(cfg, file_path)

    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["image_size"] == 256
    assert data["seed"] == 42
    assert data["score_mode"] == "global"

    assert load_config(file_path) == cfg


def test_default_yaml_matches_plan():
    cfg = load_config(REPO_ROOT / "configs" / "zeroshot.yaml")
    assert cfg.image_size == 240
    assert cfg.model == "ViT-B-16-plus-240"
    assert cfg.pretrained == "laion400m_e32"
    assert cfg.window_scales == [2, 3]
    assert cfg.num_workers == 0
    assert cfg.seed == 0


def test_invalid_score_mode_rejected():
    with pytest.raises(ValueError):
        ZeroShotConfig(score_mode="banana")
