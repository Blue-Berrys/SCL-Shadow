r"""Iterative Confidence Aggregation (ICA).

ICA estimates the *temporal* reliability of each pixel's supervision signal by
aggregating the model's own predictions across training history. For every
training sample :math:`i` it maintains an image-specific historical prediction
map :math:`\bar{P}^{(i)}`, initialised to zero and updated in place via an
exponential moving average (EMA) each time the image is sampled (Eq. (4)):

.. math::
    \bar{P}_t = \alpha\,\bar{P}_{t-1} + (1 - \alpha)\,P_{\mathrm{sh}, t}.

Pixels are then split into a high-confidence region :math:`\Omega_H` and an
uncertain region :math:`\Omega_U`, and a per-pixel weight :math:`W_{\mathrm{ICA}}`
is assigned (Eqs. (5)-(6)):

.. math::
    W_{\mathrm{ICA}}(x, y) =
    \begin{cases}
        1.0,                      & (x, y) \in \Omega_H \\
        \lambda_{\mathrm{ICA}},   & (x, y) \in \Omega_U
    \end{cases}

The historical maps are kept on CPU to avoid growing GPU memory with dataset
size; only the current batch's maps are moved to the working device.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch


class IterativeConfidenceAggregation:
    r"""Maintain per-image EMA prediction history and emit ICA weights.

    Args:
        alpha: EMA smoothing coefficient :math:`\alpha \in [0, 1)` (default 0.9;
            the paper uses 0.9). Larger values yield more stable but
            slower-adapting history maps; ``alpha=0`` disables smoothing.
        tau_high: Upper confidence threshold :math:`\tau_{\mathrm{high}}`.
        tau_low: Lower confidence threshold :math:`\tau_{\mathrm{low}}`.
        lambda_ica: Down-weighting factor :math:`\lambda_{\mathrm{ICA}} \in (0, 1)`
            applied to uncertain pixels.
        storage_device: Device on which the history maps are stored. ``"cpu"`` is
            recommended so memory scales with the dataset rather than the GPU.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        tau_high: float = 0.80,
        tau_low: float = 0.20,
        lambda_ica: float = 0.50,
        storage_device: str = "cpu",
    ) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        if not 0.0 <= tau_low < tau_high <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= tau_low < tau_high <= 1, "
                f"got tau_low={tau_low}, tau_high={tau_high}"
            )
        if not 0.0 < lambda_ica <= 1.0:
            raise ValueError(f"lambda_ica must be in (0, 1], got {lambda_ica}")

        self.alpha = alpha
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.lambda_ica = lambda_ica
        self.storage_device = torch.device(storage_device)

        # index -> historical prediction map of shape (1, H, W)
        self._history: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ utils
    def reset(self) -> None:
        """Clear all stored history maps."""
        self._history.clear()

    def state_dict(self) -> Dict[int, torch.Tensor]:
        return {k: v.clone() for k, v in self._history.items()}

    def load_state_dict(self, state: Dict[int, torch.Tensor]) -> None:
        self._history = {int(k): v.to(self.storage_device) for k, v in state.items()}

    def __len__(self) -> int:
        return len(self._history)

    # ----------------------------------------------------------------- update
    @torch.no_grad()
    def update_and_weight(
        self,
        indices: Sequence[int],
        pred_prob: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        r"""Update history with the current predictions and return ICA weights.

        Args:
            indices: Dataset indices of the samples in the batch (length ``B``).
                These key the per-image history so the same image always updates
                the same map regardless of batch composition.
            pred_prob: Shadow probability map :math:`P_{\mathrm{sh}}` after the
                sigmoid, shape ``(B, 1, H, W)`` with values in ``[0, 1]``.
            gt: Binary ground-truth mask, shape ``(B, 1, H, W)`` with values in
                ``{0, 1}``.

        Returns:
            ``W_ICA`` of shape ``(B, 1, H, W)`` on the same device as
            ``pred_prob``.
        """
        if pred_prob.dim() != 4 or pred_prob.size(1) != 1:
            raise ValueError(
                f"pred_prob must have shape (B, 1, H, W), got {tuple(pred_prob.shape)}"
            )
        if pred_prob.shape != gt.shape:
            raise ValueError(
                f"pred_prob {tuple(pred_prob.shape)} and gt {tuple(gt.shape)} "
                "must have the same shape"
            )
        if len(indices) != pred_prob.size(0):
            raise ValueError(
                f"len(indices)={len(indices)} must match batch size {pred_prob.size(0)}"
            )

        device = pred_prob.device
        pred_cpu = pred_prob.detach().to(self.storage_device)

        bar_p = torch.empty_like(pred_cpu)
        for b, idx in enumerate(indices):
            idx = int(idx)
            cur = pred_cpu[b]  # (1, H, W)
            hist = self._history.get(idx)
            if hist is None or hist.shape != cur.shape:
                # Initialised to zero: bar_P_0 = 0 => bar_P_1 = (1 - alpha) * P.
                hist = torch.zeros_like(cur)
            hist = self.alpha * hist + (1.0 - self.alpha) * cur
            self._history[idx] = hist
            bar_p[b] = hist

        bar_p = bar_p.to(device)
        return self._weight_from_history(bar_p, gt)

    def _weight_from_history(self, bar_p: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        r"""Map an aggregated history map :math:`\bar{P}` and GT to ICA weights.

        High-confidence region :math:`\Omega_H` (Eq. (5)):
        consistently-correct pixels, i.e. ``bar_p >= tau_high & gt == 1`` or
        ``bar_p <= tau_low & gt == 0``. Everything else is uncertain
        (:math:`\Omega_U`) and is down-weighted by ``lambda_ica``.
        """
        gt_bin = gt > 0.5
        high_conf = (
            (bar_p >= self.tau_high) & gt_bin
        ) | (
            (bar_p <= self.tau_low) & (~gt_bin)
        )
        weight = torch.full_like(bar_p, self.lambda_ica)
        weight[high_conf] = 1.0
        return weight
