"""SCL-Shadow: Self-Calibrated Shadow Detection with Spatial Consistency
Constraints under Noisy Labels.

This package provides a *training-time* reliability-weighting toolkit that can be
plugged on top of any shadow-detection backbone (the paper uses SDDNet). The
three core components are:

* :class:`scl.calibration.ica.IterativeConfidenceAggregation` (ICA) --
  temporal reliability from per-image EMA prediction history.
* :class:`scl.calibration.scv.SpatialConsistencyVerification` (SCV) --
  spatial reliability from local boundary-structure agreement.
* :class:`scl.calibration.brlf.BoundaryRefinedLoss` (BRLF) -- a warm-up +
  per-epoch alternation schedule between original and weighted supervision.

None of these components modify the raw annotations, and none of them change the
inference network, so the deployed model is identical to the chosen backbone.
"""

from scl.calibration import (
    BoundaryRefinedLoss,
    IterativeConfidenceAggregation,
    SpatialConsistencyVerification,
    combined_weight,
)
from scl.metrics import BERMeter

__all__ = [
    "IterativeConfidenceAggregation",
    "SpatialConsistencyVerification",
    "BoundaryRefinedLoss",
    "combined_weight",
    "BERMeter",
]

__version__ = "0.1.0"
