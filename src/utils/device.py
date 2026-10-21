"""Pick the best available PyTorch device (cuda -> mps -> cpu)."""
from __future__ import annotations

import os
from typing import Optional

import torch


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def _use_mps() -> torch.device:
    # Let unsupported MPS ops silently fall back to CPU instead of crashing.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return torch.device("mps")


def get_device(forced_device: Optional[str] = None) -> torch.device:
    """Return a torch.device.

    forced_device: None/"auto" -> auto-select cuda > mps > cpu.
    "cuda"/"mps" -> used if available, otherwise falls back to cpu.
    """
    choice = (forced_device or "auto").lower()

    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if choice == "mps":
        return _use_mps() if _mps_available() else torch.device("cpu")
    if choice != "auto":
        raise ValueError(f"Unknown device '{forced_device}'. Use auto, cpu, cuda or mps.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return _use_mps()
    return torch.device("cpu")
