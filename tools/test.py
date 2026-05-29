#!/usr/bin/env python
"""Evaluate a trained SCL-Shadow checkpoint and report BER.

Example::

    python tools/test.py --config configs/sbu.yaml \
        --checkpoint checkpoints/scl_best.pth --root /path/to/UCF
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scl.backbone import build_backbone
from scl.data import build_dataloader
from scl.engine import evaluate
from scl.utils import get_logger, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SCL-Shadow")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default=None, help="Override the evaluation dataset root.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger()
    device = torch.device(args.device)

    size = tuple(cfg.data.get("size", [416, 416]))
    root = args.root or cfg.data.get("val_root") or cfg.data.train_root
    loader = build_dataloader(
        root=root,
        image_dir=cfg.data.image_dir,
        mask_dir=cfg.data.mask_dir,
        size=size,
        train=False,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.get("num_workers", 4),
    )

    model = build_backbone(cfg.model.get("name", "reference_unet")).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))

    res = evaluate(model, loader, device)
    logger.info(
        f"Eval on {root}: BER {res.ber:.2f} | Shadow {res.shadow_err:.2f} | "
        f"NonShadow {res.non_shadow_err:.2f}"
    )


if __name__ == "__main__":
    main()
