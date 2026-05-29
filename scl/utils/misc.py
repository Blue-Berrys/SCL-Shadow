"""Miscellaneous training utilities: seeding, logging, LR schedule, checkpoints."""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "scl", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


class PolyLR(_LRScheduler):
    r"""Polynomial learning-rate decay.

    .. math::
        lr_t = lr_0 \left(1 - \frac{t}{T}\right)^{p}

    Matches the paper's schedule (initial lr :math:`5\times10^{-4}`, power
    ``0.7``). ``t`` and ``T`` are counted in optimizer steps.
    """

    def __init__(self, optimizer, max_steps: int, power: float = 0.7, last_epoch: int = -1):
        self.max_steps = max(1, max_steps)
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1.0 - min(self.last_epoch, self.max_steps) / self.max_steps) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
