r"""Backbone interface and a lightweight reference implementation.

SCL-Shadow is a *training-time* toolkit and is intentionally backbone-agnostic.
The paper builds on SDDNet (Style-guided Dual-layer Disentanglement Network,
ACM MM 2023). To plug in your own backbone, return a :class:`BackboneOutput`
from ``forward`` exposing:

* ``logits``  -- raw shadow logits, shape ``(B, 1, H, W)``;
* ``l_rec``   -- reconstruction-consistency loss (scalar);
* ``l_aux``   -- style-auxiliary loss (scalar).

A small :class:`ReferenceUNet` is included so the training/evaluation pipeline
runs end-to-end out of the box (e.g. for smoke tests and toy experiments). It is
**not** the network used to produce the paper's numbers -- for that, wrap the
official SDDNet (see ``SDDNetAdapter`` and the README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneOutput:
    """Standardised backbone output consumed by the training pipeline."""

    logits: torch.Tensor          # (B, 1, H, W) raw shadow logits
    l_rec: torch.Tensor           # scalar reconstruction-consistency loss
    l_aux: torch.Tensor           # scalar style-auxiliary loss

    def prob(self) -> torch.Tensor:
        """Shadow probability map after the sigmoid."""
        return torch.sigmoid(self.logits)


@runtime_checkable
class ShadowBackbone(Protocol):
    """Structural type any compatible backbone must satisfy."""

    def __call__(self, x: torch.Tensor) -> BackboneOutput: ...


# --------------------------------------------------------------------------- #
# Reference backbone (small U-Net) -- for pipeline validation, not SOTA.
# --------------------------------------------------------------------------- #
class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReferenceUNet(nn.Module):
    r"""A compact U-Net that mimics SDDNet's *interface*, not its architecture.

    To keep the reliability-weighting pipeline self-contained, this backbone
    also produces two auxiliary terms:

    * ``l_rec`` -- a feature reconstruction-consistency loss (an autoencoder-style
      reconstruction of the input from the disentangled "background" branch);
    * ``l_aux`` -- a small style-decorrelation penalty between the shadow and
      background feature branches.

    These mirror the *roles* of SDDNet's ``L_rec`` / ``L_aux`` so the BRLF
    schedule and weighting machinery can be exercised faithfully.
    """

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(3, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        # Two disentangled heads sharing the encoder bottleneck.
        self.shadow_head = _ConvBlock(base * 4, base * 4)
        self.bg_head = _ConvBlock(base * 4, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

        # Background-branch decoder for the reconstruction-consistency loss.
        self.recon = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        f_shadow = self.shadow_head(e3)
        f_bg = self.bg_head(e3)

        d2 = self.dec2(torch.cat([self.up2(f_shadow), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.out(d1)

        # L_rec: reconstruct the input from the background branch.
        recon = self.recon(f_bg)
        recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        l_rec = F.l1_loss(recon, x)

        # L_aux: decorrelate the two feature branches (encourage disentanglement).
        fs = F.normalize(f_shadow.flatten(1), dim=1)
        fb = F.normalize(f_bg.flatten(1), dim=1)
        l_aux = (fs * fb).sum(dim=1).abs().mean()

        return BackboneOutput(logits=logits, l_rec=0.1 * l_rec, l_aux=0.01 * l_aux)


class SDDNetAdapter(nn.Module):
    """Adapter skeleton for wrapping the official SDDNet.

    Clone the official SDDNet implementation (ACM MM 2023) and wire its outputs
    into :class:`BackboneOutput` here. The expected mapping is::

        logits  <- SDDNet final shadow logits (pre-sigmoid)
        l_rec   <- SDDNet reconstruction-consistency loss
        l_aux   <- SDDNet style-auxiliary loss

    See the README for the upstream repository link.
    """

    def __init__(self, sddnet: nn.Module) -> None:
        super().__init__()
        self.sddnet = sddnet

    def forward(self, x: torch.Tensor) -> BackboneOutput:  # pragma: no cover
        raise NotImplementedError(
            "Wire the official SDDNet outputs into BackboneOutput here. "
            "See the README 'Using the SDDNet backbone' section."
        )


def build_backbone(name: str = "reference_unet", **kwargs) -> nn.Module:
    """Factory for the available backbones."""
    name = name.lower()
    if name in ("reference_unet", "unet", "reference"):
        return ReferenceUNet(**kwargs)
    raise ValueError(
        f"Unknown backbone '{name}'. Use 'reference_unet', or wrap the official "
        "SDDNet via SDDNetAdapter."
    )
