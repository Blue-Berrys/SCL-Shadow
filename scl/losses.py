r"""Loss assembly for SCL-Shadow.

The backbone (e.g. SDDNet) is expected to expose three terms:

* ``L_BCE``  -- pixel-wise binary cross-entropy on the shadow probability map;
* ``L_rec``  -- reconstruction-consistency loss of the disentangled features;
* ``L_aux``  -- style-auxiliary loss.

Following the paper, the reliability weighting is applied **only** to the BCE
term so that the structure-preserving terms ``L_rec`` and ``L_aux`` keep the
backbone's disentanglement stable:

* Original objective (Eq. (3)):
  :math:`\mathcal{L}_{\text{Total\_SDD}} = L_{\mathrm{BCE}} + L_{\mathrm{rec}} + L_{\mathrm{aux}}`.
* Weighted objective (Eq. (10)):
  :math:`\mathcal{L}_{\text{Total\_W}} = \sum_{x,y} W_{\mathrm{combined}}(x, y)\,\ell_{\mathrm{BCE}}(x, y) + L_{\mathrm{rec}} + L_{\mathrm{aux}}`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def per_pixel_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""Per-pixel BCE :math:`\ell_{\mathrm{BCE}}(x, y)` from raw logits.

    Args:
        logits: Raw (pre-sigmoid) shadow logits, shape ``(B, 1, H, W)``.
        target: Binary target, shape ``(B, 1, H, W)``.

    Returns:
        Per-pixel loss of shape ``(B, 1, H, W)`` (no reduction).
    """
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none")


def weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> torch.Tensor:
    r"""Reliability-weighted BCE.

    Args:
        logits: Raw shadow logits, shape ``(B, 1, H, W)``.
        target: Binary target, shape ``(B, 1, H, W)``.
        weight: Per-pixel reliability weight :math:`W_{\mathrm{combined}}`. If
            ``None``, this reduces to the standard (mean) BCE.
        normalize: If ``True``, divide by the number of elements so the scale is
            comparable to the unweighted mean BCE (recommended for stable
            optimisation). If ``False``, return the raw weighted sum as written
            in Eq. (10).

    Returns:
        Scalar loss.
    """
    loss = per_pixel_bce(logits, target)
    if weight is not None:
        loss = loss * weight
    if normalize:
        return loss.mean()
    return loss.sum()


def total_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    l_rec: torch.Tensor,
    l_aux: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> torch.Tensor:
    r"""Assemble the full objective.

    When ``weight is None`` this returns :math:`\mathcal{L}_{\text{Total\_SDD}}`;
    otherwise it returns :math:`\mathcal{L}_{\text{Total\_W}}` (weighting only the
    BCE term). ``l_rec`` and ``l_aux`` are added unchanged in both cases.
    """
    bce = weighted_bce(logits, target, weight=weight, normalize=normalize)
    return bce + l_rec + l_aux
