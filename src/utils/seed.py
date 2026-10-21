"""Deterministic seeding across random, numpy and torch (CPU, CUDA, MPS)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
