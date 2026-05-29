"""Evaluation loop computing BER over a dataloader."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from scl.backbone import BackboneOutput
from scl.metrics import BERMeter, BERResult


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> BERResult:
    """Run inference over ``loader`` and return the BER result.

    The reliability-weighting modules (ICA/SCV/BRLF) are training-only, so
    evaluation is just a forward pass of the backbone followed by BER counting.
    """
    model.eval()
    meter = BERMeter()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        out = model(images)
        prob = out.prob() if isinstance(out, BackboneOutput) else torch.sigmoid(out)
        meter.update(prob, masks)
    return meter.compute()
