r"""Spatial Consistency Verification (SCV).

SCV estimates the *spatial* reliability of supervision by comparing the local
boundary structure of the (binarised) prediction with that of the annotation.

Pipeline (Eqs. (7)-(8) in the paper):

1. Binarise the predicted probability map at 0.5 to obtain :math:`\hat{M}`.
2. For every non-overlapping :math:`2\times2` block, the boundary indicator
   :math:`\rho` is 1 iff the block contains both shadow and non-shadow pixels.
3. For each central block, count boundary blocks over its local neighbourhood
   (the central block plus its four diagonal corner blocks) for both prediction
   and ground truth:

   .. math::
       S^{\mathrm{pred}}_i = \sum_{k=0}^{4}\rho(\hat{M}_{i,k}), \qquad
       S^{\mathrm{gt}}_i   = \sum_{k=0}^{4}\rho(G_{i,k}).

4. Assign a block weight by boundary-count agreement:

   .. math::
       \beta(B_{i,0}) =
       \begin{cases}
           1.0,                     & |S^{\mathrm{pred}}_i - S^{\mathrm{gt}}_i| \le \epsilon \\
           \lambda_{\mathrm{SCV}},  & \text{otherwise}
       \end{cases}

5. Broadcast each block weight back to its :math:`2\times2` pixels to obtain the
   pixel-level weight map :math:`W_{\mathrm{SCV}}`.

The whole computation is fully vectorised (no Python loops over blocks).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SpatialConsistencyVerification:
    r"""Compute the SCV pixel-weight map from a prediction and annotation.

    Args:
        epsilon: Tolerance threshold :math:`\epsilon` on the boundary-count
            difference. ``epsilon=1`` tolerates minor soft-boundary variation
            while still flagging genuine structural disagreement.
        lambda_scv: Down-weighting factor :math:`\lambda_{\mathrm{SCV}} \in (0, 1)`
            applied to spatially inconsistent blocks.
        bin_threshold: Threshold used to binarise the prediction (default 0.5).
    """

    def __init__(
        self,
        epsilon: int = 1,
        lambda_scv: float = 0.60,
        bin_threshold: float = 0.5,
    ) -> None:
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        if not 0.0 < lambda_scv <= 1.0:
            raise ValueError(f"lambda_scv must be in (0, 1], got {lambda_scv}")
        self.epsilon = epsilon
        self.lambda_scv = lambda_scv
        self.bin_threshold = bin_threshold

    # --------------------------------------------------------------- internals
    @staticmethod
    def _boundary_map(mask: torch.Tensor) -> torch.Tensor:
        r"""Per-:math:`2\times2`-block boundary indicator :math:`\rho`.

        Args:
            mask: Binary mask, shape ``(B, 1, H, W)``.

        Returns:
            Float tensor of shape ``(B, 1, H // 2, W // 2)`` with values in
            ``{0, 1}``; 1 where the block mixes shadow and non-shadow pixels.
        """
        # Sum of the four pixels in each non-overlapping 2x2 block.
        block_sum = F.avg_pool2d(mask, kernel_size=2, stride=2) * 4.0
        # Boundary iff not all-zero and not all-one (i.e. 0 < sum < 4).
        rho = (block_sum > 0.5) & (block_sum < 3.5)
        return rho.float()

    @staticmethod
    def _diagonal_neighbourhood_count(rho: torch.Tensor) -> torch.Tensor:
        r"""Sum :math:`\rho` over the centre block and its four diagonal corners.

        For a central block at :math:`(p, q)` this returns
        :math:`\rho_{p,q} + \rho_{p-1,q-1} + \rho_{p-1,q+1} + \rho_{p+1,q-1}
        + \rho_{p+1,q+1}`, matching the five-term sum in Eq. (7). Borders are
        zero-padded, so out-of-range corners contribute 0.
        """
        padded = F.pad(rho, (1, 1, 1, 1), mode="constant", value=0.0)
        centre = rho
        tl = padded[:, :, 0:-2, 0:-2]
        tr = padded[:, :, 0:-2, 2:]
        bl = padded[:, :, 2:, 0:-2]
        br = padded[:, :, 2:, 2:]
        return centre + tl + tr + bl + br

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def weight(self, pred_prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        r"""Return the SCV pixel-weight map.

        Args:
            pred_prob: Shadow probability map, shape ``(B, 1, H, W)`` in ``[0, 1]``.
            gt: Binary ground-truth mask, shape ``(B, 1, H, W)``.

        Returns:
            ``W_SCV`` of shape ``(B, 1, H, W)`` on the same device as the inputs.
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
        if pred_prob.size(2) % 2 != 0 or pred_prob.size(3) % 2 != 0:
            raise ValueError(
                "SCV requires even spatial dimensions for 2x2 blocking, got "
                f"{tuple(pred_prob.shape[2:])}"
            )

        pred_mask = (pred_prob >= self.bin_threshold).float()
        gt_mask = (gt > 0.5).float()

        rho_pred = self._boundary_map(pred_mask)
        rho_gt = self._boundary_map(gt_mask)

        s_pred = self._diagonal_neighbourhood_count(rho_pred)
        s_gt = self._diagonal_neighbourhood_count(rho_gt)

        consistent = (s_pred - s_gt).abs() <= self.epsilon
        block_weight = torch.where(
            consistent,
            torch.ones_like(s_pred),
            torch.full_like(s_pred, self.lambda_scv),
        )

        # Broadcast each block weight back to its 2x2 pixels.
        return F.interpolate(block_weight, scale_factor=2, mode="nearest")
