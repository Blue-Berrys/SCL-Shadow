r"""Training engine implementing the BRLF-scheduled, reliability-weighted loop.

Per training step (see Algorithm 1 in the paper):

1. Forward the backbone to obtain shadow logits and the auxiliary terms
   ``l_rec`` / ``l_aux``.
2. Update ICA history with the current probability map and read back
   ``W_ICA``; compute ``W_SCV`` from local boundary agreement.
3. Decide via BRLF whether this epoch uses the original or weighted loss.
4. Assemble the objective (weighting only the BCE term) and optimise.

The ICA/SCV/BRLF machinery is active only during training; inference uses the
bare backbone, so the deployed model has no extra parameters or latency.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from scl.backbone import BackboneOutput
from scl.calibration import (
    BoundaryRefinedLoss,
    IterativeConfidenceAggregation,
    SpatialConsistencyVerification,
    combined_weight,
)
from scl.engine.evaluator import evaluate
from scl.losses import total_loss
from scl.metrics import BERResult
from scl.utils.misc import PolyLR, get_logger, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        ica: IterativeConfidenceAggregation,
        scv: SpatialConsistencyVerification,
        brlf: BoundaryRefinedLoss,
        device: torch.device,
        max_epochs: int = 50,
        lr: float = 5e-4,
        poly_power: float = 0.7,
        ckpt_path: str = "checkpoints/scl_best.pth",
        log_interval: int = 20,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ica = ica
        self.scv = scv
        self.brlf = brlf
        self.device = device
        self.max_epochs = max_epochs
        self.ckpt_path = ckpt_path
        self.log_interval = log_interval
        self.logger = get_logger()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        steps_per_epoch = max(1, len(train_loader))
        self.scheduler = PolyLR(
            self.optimizer, max_steps=max_epochs * steps_per_epoch, power=poly_power
        )
        self.best_ber = float("inf")

    # ------------------------------------------------------------------ epoch
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        use_weighted = self.brlf.use_weighted(epoch)
        phase = self.brlf.phase_name(epoch)
        running = 0.0

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            indices = batch["index"]

            out = self.model(images)
            if not isinstance(out, BackboneOutput):
                raise TypeError("Backbone must return a BackboneOutput.")
            prob = out.prob()

            # Always update ICA history so the EMA keeps tracking, even on
            # original-loss epochs; only *apply* the weights when BRLF says so.
            w_ica = self.ica.update_and_weight(indices, prob, masks)
            weight = None
            if use_weighted:
                w_scv = self.scv.weight(prob, masks)
                weight = combined_weight(w_ica, w_scv).detach()

            loss = total_loss(
                out.logits, masks, out.l_rec, out.l_aux, weight=weight, normalize=True
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running += float(loss.detach())
            if step % self.log_interval == 0:
                self.logger.info(
                    f"epoch {epoch:03d} [{phase:>8}] step {step:04d}/"
                    f"{len(self.train_loader)} loss {float(loss):.4f} "
                    f"lr {self.scheduler.get_last_lr()[0]:.2e}"
                )

        return running / max(1, len(self.train_loader))

    # -------------------------------------------------------------------- fit
    def fit(self) -> float:
        for epoch in range(1, self.max_epochs + 1):
            avg_loss = self.train_one_epoch(epoch)
            msg = f"epoch {epoch:03d} done | avg_loss {avg_loss:.4f}"

            if self.val_loader is not None:
                res: BERResult = evaluate(self.model, self.val_loader, self.device)
                msg += (
                    f" | BER {res.ber:.2f} Shad {res.shadow_err:.2f} "
                    f"NoShad {res.non_shadow_err:.2f}"
                )
                if res.ber < self.best_ber:
                    self.best_ber = res.ber
                    save_checkpoint(
                        self.ckpt_path, self.model, self.optimizer, epoch,
                        extra={"ber": res.ber},
                    )
                    msg += f"  (best -> {self.ckpt_path})"
            self.logger.info(msg)

        self.logger.info(f"training finished. best BER: {self.best_ber:.2f}")
        return self.best_ber
