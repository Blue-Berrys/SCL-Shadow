"""Unit tests for the ICA, SCV, and BRLF components.

Run with::

    pytest -q

These tests check the components against the equations in the paper rather than
training behaviour, so they run in milliseconds on CPU.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scl.calibration import (
    BoundaryRefinedLoss,
    IterativeConfidenceAggregation,
    SpatialConsistencyVerification,
    combined_weight,
)


# --------------------------------------------------------------------- ICA
def test_ica_ema_initialisation():
    """First update with zero-initialised history gives (1-alpha)*P."""
    ica = IterativeConfidenceAggregation(alpha=0.9)
    pred = torch.full((1, 1, 4, 4), 0.8)
    gt = torch.ones(1, 1, 4, 4)
    ica.update_and_weight([0], pred, gt)
    hist = ica.state_dict()[0]
    assert torch.allclose(hist, torch.full_like(hist, (1 - 0.9) * 0.8), atol=1e-6)


def test_ica_high_vs_uncertain_weights():
    """Confident-correct pixels get weight 1; the rest get lambda_ica."""
    ica = IterativeConfidenceAggregation(
        alpha=0.0, tau_high=0.8, tau_low=0.2, lambda_ica=0.5
    )  # alpha=0 -> history equals current prediction
    pred = torch.tensor([[[[0.9, 0.5], [0.1, 0.5]]]])
    gt = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    w = ica.update_and_weight([0], pred, gt)
    # (0,0): pred>=0.8 & gt==1 -> high (1.0)
    # (0,1): 0.5 in (0.2,0.8) -> uncertain (0.5)
    # (1,0): pred<=0.2 & gt==0 -> high (1.0)
    # (1,1): 0.5 & gt==0 -> uncertain (0.5)
    expected = torch.tensor([[[[1.0, 0.5], [1.0, 0.5]]]])
    assert torch.allclose(w, expected)


def test_ica_persistent_disagreement_is_uncertain():
    """Confident-but-wrong pixels are uncertain, not high-confidence."""
    ica = IterativeConfidenceAggregation(alpha=0.0, lambda_ica=0.3)
    pred = torch.tensor([[[[0.95]]]])  # very confident shadow
    gt = torch.tensor([[[[0.0]]]])     # but label says non-shadow
    w = ica.update_and_weight([0], pred, gt)
    assert torch.allclose(w, torch.tensor([[[[0.3]]]]))


# --------------------------------------------------------------------- SCV
def test_scv_identical_pred_and_gt_full_weight():
    """When prediction equals GT, all boundary counts agree -> weight 1."""
    scv = SpatialConsistencyVerification(epsilon=1, lambda_scv=0.6)
    gt = torch.zeros(1, 1, 8, 8)
    gt[:, :, :, 4:] = 1.0  # a clean vertical boundary
    w = scv.weight(gt, gt)
    assert torch.allclose(w, torch.ones_like(w))


def test_scv_downweights_inconsistent_region():
    """A fragmented prediction vs. a clean GT should trigger down-weighting.

    The GT has no boundaries (S_gt = 0 everywhere); the speckle creates a local
    boundary block, so |S_pred - S_gt| = 1. With a strict tolerance (epsilon=0)
    this exceeds the threshold and the region is down-weighted.
    """
    scv = SpatialConsistencyVerification(epsilon=0, lambda_scv=0.6)
    gt = torch.zeros(1, 1, 8, 8)            # no boundaries at all
    pred = torch.zeros(1, 1, 8, 8)
    # Inject an isolated speckle to create a local boundary in the prediction.
    pred[:, :, 3, 3] = 1.0
    w = scv.weight(pred, gt)
    assert torch.isclose(w.min(), torch.tensor(0.6))
    assert w.shape == pred.shape


def test_scv_requires_even_dims():
    scv = SpatialConsistencyVerification()
    try:
        scv.weight(torch.zeros(1, 1, 7, 8), torch.zeros(1, 1, 7, 8))
    except ValueError:
        return
    raise AssertionError("SCV should reject odd spatial dimensions")


# -------------------------------------------------------------------- BRLF
def test_brlf_warmup_uses_original():
    brlf = BoundaryRefinedLoss(warmup_epochs=5, period=1)
    for e in range(1, 6):
        assert brlf.use_weighted(e) is False
        assert brlf.phase_name(e) == "warmup"


def test_brlf_alternation_after_warmup():
    """Eq. (11): after warm-up, weighted on odd epochs, original on even."""
    brlf = BoundaryRefinedLoss(warmup_epochs=5, period=1)
    assert brlf.use_weighted(6) is False   # even -> original
    assert brlf.use_weighted(7) is True    # odd  -> weighted
    assert brlf.use_weighted(8) is False
    assert brlf.use_weighted(9) is True


# ---------------------------------------------------------------- combined
def test_combined_weight_is_gate():
    w_ica = torch.tensor([[[[1.0, 0.5]]]])
    w_scv = torch.tensor([[[[0.6, 1.0]]]])
    out = combined_weight(w_ica, w_scv)
    assert torch.allclose(out, torch.tensor([[[[0.6, 0.5]]]]))
