"""WinCLIP compositional prompt ensemble with ablation modes."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

VALID_PROMPT_MODES = ("single", "state_only", "full_ensemble")

CLASS_NAMES = {
    "carpet": "carpet", "leather": "leather", "tile": "tile", "bottle": "bottle",
    "transistor": "transistor", "metal_nut": "metal nut", "screw": "screw", "pill": "pill",
    "toothbrush": "toothbrush", "zipper": "zipper", "grid": "grid", "wood": "wood",
    "cable": "cable", "capsule": "capsule", "hazelnut": "hazelnut",
}

NORMAL_STATES = [
    "{}", "flawless {}", "perfect {}", "unblemished {}",
    "{} without flaw", "{} without defect", "{} without damage",
]
ANOMALOUS_STATES = ["damaged {}", "broken {}", "{} with flaw", "{} with defect", "{} with damage"]

TEMPLATES = [
    "a photo of a {}.", "a cropped photo of the {}.", "a close-up photo of a {}.",
    "a bright photo of a {}.", "a dark photo of the {}.", "a blurry photo of the {}.",
    "a jpeg corrupted photo of the {}.", "a photo of the small {}.", "a photo of the large {}.",
    "there is a {} in the scene.", "this is one {} in the scene.",
    "a photo of a {} for visual inspection.", "a photo of the {} for anomaly detection.",
]


def build_prompts(class_name: str, prompt_mode: str = "full_ensemble") -> tuple[list[str], list[str]]:
    """Return (normal_prompts, anomalous_prompts) for the given ablation mode."""
    if prompt_mode not in VALID_PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of {VALID_PROMPT_MODES}, got '{prompt_mode}'")
    if prompt_mode == "single":
        templates, normal, anomalous = TEMPLATES[:1], ["{}"], ["damaged {}"]
    elif prompt_mode == "state_only":
        templates, normal, anomalous = TEMPLATES[:1], NORMAL_STATES, ANOMALOUS_STATES
    else:
        templates, normal, anomalous = TEMPLATES, NORMAL_STATES, ANOMALOUS_STATES

    def expand(states: list[str]) -> list[str]:
        return [t.format(s.format(class_name)) for s in states for t in templates]

    return expand(normal), expand(anomalous)


@torch.no_grad()
def build_text_embedding(
    model,
    tokenizer,
    category: str,
    device: torch.device | str,
    prompt_mode: str = "full_ensemble",
    cache_dir: str | Path | None = None,
    pretrained_tag: str | None = None,
) -> Tensor:
    """Return [2, D] unit-norm text embeddings (row 0 normal, row 1 anomalous).

    Cached at <cache_dir>/text_emb_<category>_<pretrained_tag>_<prompt_mode>.pt when both
    cache_dir and pretrained_tag are given (random-weight models are never cached).
    """
    cache_path = None
    if cache_dir is not None and pretrained_tag:
        cache_path = Path(cache_dir) / f"text_emb_{category}_{pretrained_tag}_{prompt_mode}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location=device)

    class_name = CLASS_NAMES.get(category, category.replace("_", " "))
    rows = []
    for prompts in build_prompts(class_name, prompt_mode):
        tokens = tokenizer(prompts).to(device)
        emb = F.normalize(model.encode_text(tokens).float(), dim=-1)
        rows.append(F.normalize(emb.mean(dim=0), dim=-1))
    text = torch.stack(rows, dim=0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(text.cpu(), cache_path)
    return text
