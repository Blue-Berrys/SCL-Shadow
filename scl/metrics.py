r"""Evaluation metric: Balanced Error Rate (BER).

BER is the standard shadow-detection metric (lower is better):

.. math::
    BER = \left(1 - \frac{1}{2}\left(\frac{N_{tp}}{N_p} + \frac{N_{tn}}{N_n}\right)\right)\times 100,

where :math:`N_{tp}`, :math:`N_{tn}`, :math:`N_p`, :math:`N_n` are the true
positive, true negative, total shadow (positive) and total non-shadow (negative)
pixel counts. The shadow and non-shadow region error rates are reported
alongside BER.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BERResult:
    ber: float
    shadow_err: float
    non_shadow_err: float

    def as_dict(self) -> dict:
        return {
            "BER": self.ber,
            "Shadow": self.shadow_err,
            "NonShadow": self.non_shadow_err,
        }


class BERMeter:
    """Accumulate pixel counts across a dataset and compute BER.

    Predictions and targets are matched at the evaluation resolution. By
    convention the prediction is binarised at 0.5.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.n_tp = 0.0
        self.n_tn = 0.0
        self.n_p = 0.0
        self.n_n = 0.0

    @torch.no_grad()
    def update(self, pred_prob: torch.Tensor, gt: torch.Tensor) -> None:
        """Accumulate counts for a batch.

        Args:
            pred_prob: Shadow probability map in ``[0, 1]``, shape ``(B, 1, H, W)``.
            gt: Binary mask, shape ``(B, 1, H, W)``.
        """
        pred = (pred_prob >= self.threshold)
        target = (gt > 0.5)
        pos = target
        neg = ~target

        self.n_tp += float((pred & pos).sum())
        self.n_tn += float((~pred & neg).sum())
        self.n_p += float(pos.sum())
        self.n_n += float(neg.sum())

    def compute(self) -> BERResult:
        eps = 1e-8
        tpr = self.n_tp / (self.n_p + eps)  # recall on shadow
        tnr = self.n_tn / (self.n_n + eps)  # recall on non-shadow
        ber = (1.0 - 0.5 * (tpr + tnr)) * 100.0
        shadow_err = (1.0 - tpr) * 100.0
        non_shadow_err = (1.0 - tnr) * 100.0
        return BERResult(ber=ber, shadow_err=shadow_err, non_shadow_err=non_shadow_err)
