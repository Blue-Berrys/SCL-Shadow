#!/usr/bin/env python
"""Train SCL-Shadow on a shadow-detection benchmark.

Example::

    python tools/train.py --config configs/sbu.yaml

All hyperparameters come from the YAML config; CLI flags override a few common
ones. See ``configs/*.yaml`` for the full set.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scl.backbone import build_backbone
from scl.calibration import (
    BoundaryRefinedLoss,
    IterativeConfidenceAggregation,
    SpatialConsistencyVerification,
)
from scl.data import build_dataloader
from scl.engine import Trainer
from scl.utils import get_logger, load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SCL-Shadow")
    p.add_argument("--config", required=True, help="Path to a YAML config file.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    p.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger()

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    epochs = args.epochs or cfg.train.epochs
    batch_size = args.batch_size or cfg.train.batch_size
    size = tuple(cfg.data.get("size", [416, 416]))
    device = torch.device(args.device)

    train_loader = build_dataloader(
        root=cfg.data.train_root,
        image_dir=cfg.data.image_dir,
        mask_dir=cfg.data.mask_dir,
        size=size,
        train=True,
        batch_size=batch_size,
        num_workers=cfg.data.get("num_workers", 4),
    )
    val_loader = None
    if cfg.data.get("val_root"):
        val_loader = build_dataloader(
            root=cfg.data.val_root,
            image_dir=cfg.data.image_dir,
            mask_dir=cfg.data.mask_dir,
            size=size,
            train=False,
            batch_size=batch_size,
            num_workers=cfg.data.get("num_workers", 4),
        )

    model = build_backbone(cfg.model.get("name", "reference_unet"))

    ica = IterativeConfidenceAggregation(
        alpha=cfg.ica.alpha,
        tau_high=cfg.ica.tau_high,
        tau_low=cfg.ica.tau_low,
        lambda_ica=cfg.ica.lambda_ica,
    )
    scv = SpatialConsistencyVerification(
        epsilon=cfg.scv.epsilon,
        lambda_scv=cfg.scv.lambda_scv,
    )
    brlf = BoundaryRefinedLoss(
        warmup_epochs=cfg.brlf.warmup_epochs,
        period=cfg.brlf.get("period", 1),
    )

    logger.info(f"Backbone: {cfg.model.get('name', 'reference_unet')} | device: {device}")
    logger.info(
        f"ICA(alpha={ica.alpha}, tau=[{ica.tau_low},{ica.tau_high}], "
        f"lambda={ica.lambda_ica}) | SCV(eps={scv.epsilon}, lambda={scv.lambda_scv}) | "
        f"BRLF(warmup={brlf.warmup_epochs}, period={brlf.period})"
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        ica=ica,
        scv=scv,
        brlf=brlf,
        device=device,
        max_epochs=epochs,
        lr=cfg.train.lr,
        poly_power=cfg.train.get("poly_power", 0.7),
        ckpt_path=cfg.train.get("ckpt_path", "checkpoints/scl_best.pth"),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
