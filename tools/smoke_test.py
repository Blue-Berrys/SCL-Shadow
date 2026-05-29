#!/usr/bin/env python
"""End-to-end smoke test on synthetic data (no datasets required).

Generates a tiny synthetic shadow dataset in a temp directory, runs a few
training epochs with the reference backbone, and evaluates BER. Useful for
verifying the pipeline wiring before downloading the real benchmarks.

Example::

    python tools/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scl.backbone import build_backbone
from scl.calibration import (
    BoundaryRefinedLoss,
    IterativeConfidenceAggregation,
    SpatialConsistencyVerification,
)
from scl.data import build_dataloader
from scl.engine import Trainer, evaluate
from scl.utils import get_logger, set_seed


def make_synthetic_split(root: str, n: int = 8, size: int = 64) -> None:
    img_dir = os.path.join(root, "images")
    msk_dir = os.path.join(root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        mask = np.zeros((size, size), dtype=np.uint8)
        x0 = rng.integers(0, size // 2)
        y0 = rng.integers(0, size // 2)
        mask[y0:y0 + size // 3, x0:x0 + size // 3] = 255
        img = np.stack([255 - mask, rng.integers(0, 60, (size, size), dtype=np.uint8),
                        mask], axis=-1).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(img_dir, f"{i:04d}.jpg"))
        Image.fromarray(mask).save(os.path.join(msk_dir, f"{i:04d}.png"))


def main() -> None:
    logger = get_logger()
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as tmp:
        train_root = os.path.join(tmp, "train")
        val_root = os.path.join(tmp, "val")
        make_synthetic_split(train_root, n=8)
        make_synthetic_split(val_root, n=4)

        train_loader = build_dataloader(
            train_root, size=(64, 64), train=True, batch_size=2, num_workers=0
        )
        val_loader = build_dataloader(
            val_root, size=(64, 64), train=False, batch_size=2, num_workers=0
        )

        model = build_backbone("reference_unet", base=8)
        ica = IterativeConfidenceAggregation()
        scv = SpatialConsistencyVerification()
        brlf = BoundaryRefinedLoss(warmup_epochs=1)

        trainer = Trainer(
            model, train_loader, val_loader, ica, scv, brlf, device,
            max_epochs=4, ckpt_path=os.path.join(tmp, "ck.pth"), log_interval=2,
        )
        trainer.fit()

        res = evaluate(model, val_loader, device)
        logger.info(f"Smoke test final BER: {res.ber:.2f}")
        logger.info("Smoke test PASSED (pipeline runs end-to-end).")


if __name__ == "__main__":
    main()
