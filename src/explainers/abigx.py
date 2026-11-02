from typing import Any
import torch
from torch import Tensor
from captum.attr import IntegratedGradients

from src.detectors.base import BaseDetector
from src.explainers.base import BaseExplainer, DifferentiableScoreWrapper

# CLIP normalized bounds for RGB in [0, 1]
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ABIGXExplainer(BaseExplainer):
    """ABIGX-Style Explainer: Adversarial Fault Reconstruction (AFR) + Integrated Gradients.

    Stage 1 (AFR): Uses Projected Gradient Descent (PGD) to reconstruct a normal-looking
                   counterfactual baseline x_AFR from the anomalous input x.
    Stage 2 (IG):  Accumulates gradients along the linear path from x_AFR to x.
    Fallback:      If AFR fails to reduce the anomaly score sufficiently, falls back
                   to Expected-IG over normal reference images (or zero baseline).
    """

    def __init__(
        self,
        detector: BaseDetector,
        device: torch.device,
        afr_steps: int = 15,
        afr_step_size: float = 0.03,
        afr_eps: float = 0.35,
        afr_lambda_prox: float = 0.05,
        normal_score_threshold: float = 0.35,
        min_relative_drop: float = 0.15,
        ig_steps: int = 25,
        score_mode: str = "combined_smooth",
    ) -> None:
        super().__init__(detector=detector, device=device)
        self.afr_steps = afr_steps
        self.afr_step_size = afr_step_size
        self.afr_eps = afr_eps
        self.afr_lambda_prox = afr_lambda_prox
        self.normal_score_threshold = normal_score_threshold
        self.min_relative_drop = min_relative_drop
        self.ig_steps = ig_steps
        self.wrapper = DifferentiableScoreWrapper(detector, target_score_mode=score_mode).to(device)
        self.ig = IntegratedGradients(self.wrapper)

        mean = torch.tensor(CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.min_val = (0.0 - mean) / std
        self.max_val = (1.0 - mean) / std

    def _run_afr(self, x0: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """Runs PGD to find a normal counterfactual baseline x_AFR near x0."""
        x_curr = x0.clone().detach()
        trajectory_scores: list[float] = []

        best_x = x_curr.clone()
        best_score = float("inf")
        init_score: float | None = None

        for step_idx in range(self.afr_steps):
            x_curr = x_curr.detach().requires_grad_(True)
            score = self.wrapper(x_curr).sum()
            curr_val = float(score.detach().item())

            if step_idx == 0:
                init_score = curr_val
                best_score = curr_val
            else:
                if curr_val < best_score:
                    best_score = curr_val
                    best_x = x_curr.detach().clone()
                if curr_val <= self.normal_score_threshold:
                    trajectory_scores.append(curr_val)
                    break

            trajectory_scores.append(curr_val)

            prox_loss = torch.mean((x_curr - x0) ** 2)
            total_loss = score + self.afr_lambda_prox * prox_loss

            grad = torch.autograd.grad(total_loss, x_curr, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                x_next = x_curr - self.afr_step_size * torch.sign(grad)
                # Project onto L_inf epsilon-ball around x0
                delta = torch.clamp(x_next - x0, min=-self.afr_eps, max=self.afr_eps)
                x_curr = torch.clamp(x0 + delta, min=self.min_val, max=self.max_val)

            # Explicitly empty MPS cache between PGD steps to prevent graph buffer accumulation
            if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Evaluate final updated x_curr once if loop completed without early break
        with torch.no_grad():
            final_val = float(self.wrapper(x_curr).item())
        if init_score is None:
            init_score = final_val
            best_score = final_val
            trajectory_scores.append(final_val)
        elif len(trajectory_scores) == self.afr_steps:
            trajectory_scores.append(final_val)
            if final_val < best_score:
                best_score = final_val
                best_x = x_curr.detach().clone()

        rel_drop = (init_score - best_score) / max(abs(init_score), 1e-6)
        converged = (best_score <= self.normal_score_threshold) or (rel_drop >= self.min_relative_drop)

        meta = {
            "initial_score": init_score,
            "baseline_score": best_score,
            "relative_drop": float(rel_drop),
            "converged": bool(converged),
            "steps_run": len(trajectory_scores) - 1,
            "trajectory_scores": trajectory_scores,
        }
        return best_x.detach(), meta

    def explain(
        self,
        image: Tensor,
        reference_normals: Tensor | None = None,
    ) -> dict[str, Any]:
        """Computes ABIGX attribution map [1, H, W] and returns baseline + convergence metadata."""
        x0 = image.to(self.device, dtype=torch.float32)
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        afr_baseline, meta = self._run_afr(x0)

        if meta["converged"]:
            meta["baseline_type"] = "afr_reconstructed"
            raw_attr = self.ig.attribute(
                x0,
                baselines=afr_baseline,
                n_steps=self.ig_steps,
                internal_batch_size=1,
            )
            used_baseline = afr_baseline
        else:
            # Proposal Section 4.4 Mitigation: Fallback to Expected-IG on normal baselines
            if reference_normals is not None and reference_normals.numel() > 0:
                meta["baseline_type"] = "fallback_expected_ig"
                refs = reference_normals.to(self.device, dtype=torch.float32)
                attrs = []
                for k in range(refs.shape[0]):
                    base_k = refs[k : k + 1]
                    attr_k = self.ig.attribute(
                        x0,
                        baselines=base_k,
                        n_steps=max(2, self.ig_steps // 2),
                        internal_batch_size=1,
                    )
                    attrs.append(attr_k)
                    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                raw_attr = torch.stack(attrs, dim=0).mean(dim=0)
                used_baseline = refs[0:1]
            else:
                meta["baseline_type"] = "fallback_black_ig"
                used_baseline = torch.zeros_like(x0)
                raw_attr = self.ig.attribute(
                    x0,
                    baselines=used_baseline,
                    n_steps=self.ig_steps,
                    internal_batch_size=1,
                )

        if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Channel aggregation via L2 norm -> [1, H, W]
        spatial_map = torch.linalg.vector_norm(raw_attr, ord=2, dim=1)
        b_min = spatial_map.amin(dim=(1, 2), keepdim=True)
        b_max = spatial_map.amax(dim=(1, 2), keepdim=True)
        norm_map = (spatial_map - b_min) / (b_max - b_min + 1e-8)

        return {
            "attribution_map": norm_map.detach(),
            "raw_attribution": raw_attr.detach(),
            "baseline": used_baseline.detach(),
            "meta": meta,
        }

