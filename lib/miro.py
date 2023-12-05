import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT

from lib.core import AUModel


class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, num_channels, init=0.1, eps=1e-5, hwbc_batch=False):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.hwbc_batch = hwbc_batch

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()

        if self.hwbc_batch:
            # CLIP-ViT: [H*W+1, B, C]
            b_shape = (1, 1, self.num_channels)
        else:
            # [B, C, H, W]
            b_shape = (1, self.num_channels, 1, 1)

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps


class MIRO(AUModel):
    """Mutual-Information Regularization with Oracle"""
    def __init__(self, backbone: torch.nn.Module, ld: float, miro_lr=1e-5, **kwargs):
        super().__init__(backbone=backbone, **kwargs)
        self.frozen_backbone = copy.deepcopy(backbone)
        for p in self.frozen_backbone.parameters():
            p.requires_grad = False

        self.miro_lr = miro_lr
        self.ld = ld

        channels = self.backbone.feature_info.channels()

        self.mean_encoders = nn.ModuleList([
            MeanEncoder(c) for c in channels
        ])
        self.var_encoders = nn.ModuleList([
            VarianceEncoder(c) for c in channels
        ])

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        all_x = batch["img"]
        all_y = batch["multilabel"]
        feats = self.backbone(all_x)
        logit = self.heads["multilabel"](feats[-1]).squeeze()
        loss = self.heads["multilabel"].loss(logit, all_y)

        # MIRO
        with torch.no_grad():
            frozen_feats = self.frozen_backbone(all_x)

        reg_loss = 0.

        assert len(feats) == len(frozen_feats) == len(self.mean_encoders) == len(self.var_encoders)
        for f, fr_f, mean_enc, var_enc in zip(feats, frozen_feats, self.mean_encoders, self.var_encoders):
            # mutual information regularization
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - fr_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.

        loss += reg_loss * self.ld
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        assert self.optimizer_partial is not None, "Optimizer was not provided on initialization"
        assert self.scheduler_partial is not None, "Scheduler was not provided on initialization"

        parameters = [
            {"params": self.backbone.parameters()},
            {"params": self.heads["multilabel"].parameters()},
            {"params": self.mean_encoders.parameters(), "lr": self.miro_lr},
            {"params": self.var_encoders.parameters(), "lr": self.miro_lr},
        ]

        optimizer = self.optimizer_partial(parameters)
        scheduler = self.scheduler_partial(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
