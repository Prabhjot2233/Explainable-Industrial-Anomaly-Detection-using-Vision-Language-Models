import py_compile
from pathlib import Path
import pytest
import torch
from app.streamlit_app import load_input_tensor


def test_streamlit_app_compilation():
    """Verify that app/streamlit_app.py compiles without syntax errors."""
    app_path = Path("app/streamlit_app.py")
    assert app_path.exists()
    compiled = py_compile.compile(str(app_path))
    assert compiled is not None


def test_load_input_tensor_synthetic():
    inputs = {
        "uploaded_file": None,
        "selected_sample": None,
    }
    norm_t, gt_mask, label, is_normal = load_input_tensor(inputs, "carpet")
    assert norm_t.shape == (3, 240, 240)
    assert gt_mask.shape == (240, 240)
    assert label == "synthetic_crack"
    assert is_normal is False

