"""Reliability-calibration components (ICA, SCV, BRLF)."""

import torch

from scl.calibration.brlf import BoundaryRefinedLoss
from scl.calibration.ica import IterativeConfidenceAggregation
from scl.calibration.scv import SpatialConsistencyVerification


def combined_weight(w_ica: torch.Tensor, w_scv: torch.Tensor) -> torch.Tensor:
    r"""Gate-based fusion of temporal and spatial reliability.

    Implements Eq. (9) of the paper:

    .. math::
        W_{\mathrm{combined}}(x, y) = W_{\mathrm{ICA}}(x, y)\cdot W_{\mathrm{SCV}}(x, y).

    A pixel only receives full supervisory weight when it is *simultaneously*
    temporally stable (high ICA weight) and spatially consistent (high SCV
    weight); uncertainty in either dimension suppresses its gradient.
    """
    return w_ica * w_scv


__all__ = [
    "IterativeConfidenceAggregation",
    "SpatialConsistencyVerification",
    "BoundaryRefinedLoss",
    "combined_weight",
]
