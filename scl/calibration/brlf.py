r"""Boundary Refined Loss Function (BRLF) schedule.

BRLF is a training-time regulariser that prevents the model from over-relying on
the reliability weighting (and thereby over-suppressing boundary-region
gradients). After a warm-up phase it periodically alternates between the original
SDDNet loss and the reliability-weighted loss (Eq. (11)):

.. math::
    \mathcal{L}_e =
    \begin{cases}
        \mathcal{L}_{\text{Total\_SDD}}, & e \le E_{\mathrm{warm}}\ \text{or}\ e \bmod 2 = 0 \\
        \mathcal{L}_{\text{Total\_W}},   & \text{otherwise}
    \end{cases}

With the default period :math:`\mathcal{P} = 1`, after warm-up even epochs use
the original loss (reinforce disentanglement) and odd epochs use the weighted
loss (enforce reliability-weighted supervision).

This module only decides *which* objective to use at a given epoch; the actual
loss terms are assembled by :mod:`scl.losses`.
"""

from __future__ import annotations


class BoundaryRefinedLoss:
    r"""Epoch-level scheduler choosing original vs. weighted supervision.

    Args:
        warmup_epochs: Number of warm-up epochs :math:`E_{\mathrm{warm}}` during
            which the original loss is always used (default 5).
        period: Alternation period :math:`\mathcal{P}` (default 1, per-epoch).
        epoch_base: ``1`` if epochs are 1-indexed (as in Algorithm 1), ``0`` for
            0-indexed loops. The parity test in Eq. (11) assumes 1-indexing.
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        period: int = 1,
        epoch_base: int = 1,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        if epoch_base not in (0, 1):
            raise ValueError(f"epoch_base must be 0 or 1, got {epoch_base}")
        self.warmup_epochs = warmup_epochs
        self.period = period
        self.epoch_base = epoch_base

    def _normalise(self, epoch: int) -> int:
        """Return a 1-indexed epoch number for the parity test."""
        return epoch + (1 if self.epoch_base == 0 else 0)

    def use_weighted(self, epoch: int) -> bool:
        r"""Whether to use the reliability-weighted loss at ``epoch``.

        Returns ``False`` (use :math:`\mathcal{L}_{\text{Total\_SDD}}`) during
        warm-up and on "original-loss" cycles; ``True`` (use
        :math:`\mathcal{L}_{\text{Total\_W}}`) otherwise.
        """
        e = self._normalise(epoch)
        if e <= self.warmup_epochs:
            return False
        # Generalised per-(self.period) alternation. The first post-warm-up
        # block (group 0) uses the original loss; weighted blocks are the
        # odd-indexed groups. With period == 1 this reduces exactly to Eq. (11):
        # weighted on odd epochs after warm-up (e.g. warmup=5 -> e=6 original,
        # e=7 weighted, e=8 original, ...).
        group = (e - self.warmup_epochs - 1) // self.period
        return group % 2 == 1

    def phase_name(self, epoch: int) -> str:
        """Human-readable phase label for logging."""
        e = self._normalise(epoch)
        if e <= self.warmup_epochs:
            return "warmup"
        return "weighted" if self.use_weighted(epoch) else "original"
