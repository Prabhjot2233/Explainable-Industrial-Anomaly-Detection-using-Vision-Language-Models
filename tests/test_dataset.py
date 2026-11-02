import numpy as np
from PIL import Image

from src.data.mvtec import MVTecDataset


def _make_fake_mvtec(root):
    cat = root / "bottle"
    for sub in ["train/good", "test/good", "test/broken", "ground_truth/broken"]:
        (cat / sub).mkdir(parents=True)
    rgb = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    rgb.save(cat / "train/good/000.png")
    rgb.save(cat / "test/good/000.png")
    rgb.save(cat / "test/broken/000.png")
    m = np.zeros((64, 64), dtype=np.uint8)
    m[10:20, 10:20] = 255
    Image.fromarray(m).save(cat / "ground_truth/broken/000_mask.png")


def test_dataset_shapes_and_labels(tmp_path):
    _make_fake_mvtec(tmp_path)
    ds = MVTecDataset(tmp_path, "bottle", split="test")
    assert len(ds) == 2
    labels = sorted(ds[i]["label"] for i in range(len(ds)))
    assert labels == [0, 1]
    for i in range(len(ds)):
        item = ds[i]
        assert item["image"].shape == (3, 240, 240)
        assert item["mask"].shape == (1, 240, 240)
        assert (item["mask"].sum() > 0) == (item["label"] == 1)
    assert len(MVTecDataset(tmp_path, "bottle", split="train")) == 1
