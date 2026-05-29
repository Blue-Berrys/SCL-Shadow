#!/usr/bin/env python
"""Zero-shot, frame-by-frame video shadow detection.

Runs an image-trained checkpoint over a folder of video frames in temporal
order, writing predicted masks. This reproduces the zero-shot ViSha application
in the paper (no video-specific fine-tuning).

Example::

    python tools/infer_video.py --config configs/sbu.yaml \
        --checkpoint checkpoints/scl_best.pth \
        --frames /path/to/visha/video1/frames --out outputs/video1
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scl.backbone import build_backbone
from scl.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from scl.utils import get_logger, load_checkpoint, load_config

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot video shadow detection")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frames", required=True, help="Directory of ordered video frames.")
    p.add_argument("--out", required=True, help="Output directory for predicted masks.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger()
    device = torch.device(args.device)
    size = tuple(cfg.data.get("size", [416, 416]))

    model = build_backbone(cfg.model.get("name", "reference_unet")).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    frames = sorted(
        f for f in os.listdir(args.frames) if os.path.splitext(f)[1].lower() in _IMG_EXTS
    )
    logger.info(f"Processing {len(frames)} frames from {args.frames}")

    for name in frames:
        img = Image.open(os.path.join(args.frames, name)).convert("RGB")
        w, h = img.size
        x = img.resize((size[1], size[0]), Image.BILINEAR)
        x = TF.normalize(TF.to_tensor(x), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)

        out = model(x)
        prob = out.prob()
        mask = (prob >= 0.5).float()[0, 0].cpu()
        mask_img = Image.fromarray((mask.numpy() * 255).astype("uint8")).resize((w, h), Image.NEAREST)
        mask_img.save(os.path.join(args.out, os.path.splitext(name)[0] + ".png"))

    logger.info(f"Saved masks to {args.out}")


if __name__ == "__main__":
    main()
