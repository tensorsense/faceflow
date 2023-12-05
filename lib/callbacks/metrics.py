from typing import Optional, Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.functional.classification import (
    binary_average_precision,
    binary_f1_score,
    binary_specificity,
    binary_recall,
    binary_precision,
)


def binary_eer(p, g, n_thresholds=51):
    thresholds = np.linspace(0, 1, n_thresholds)
    metrics = {}

    fpr = torch.stack([1.0 - binary_specificity(p, g, threshold=t) for t in thresholds])
    fnr = torch.stack([1.0 - binary_recall(p, g, threshold=t) for t in thresholds])

    idx = torch.argmin(torch.abs(fnr - fpr))
    metrics["eer"] = (fnr[idx] + fpr[idx]) / 2
    metrics["eer_thresh"] = thresholds[idx]
    return metrics


class MetricsCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n_labels = pl_module.heads["multilabel"].logits_per_class
        ds_name = trainer.val_dataloaders[dataloader_idx].dataset.name
        cls_names = trainer.val_dataloaders[dataloader_idx].dataset.cls_names
        separate_sides = trainer.val_dataloaders[dataloader_idx].dataset.separate_sides
        n_classes = len(cls_names)

        gt = batch["multilabel"]  # shape batch x n_classes x n_labels
        pred = torch.sigmoid(outputs["logits"]["multilabel"])
        if not separate_sides and pred.shape[1] == n_classes * 2:
            # merge left and right AU when train is on separate and val is on joined sides
            pred = pred.view(pred.shape[0], -1, 2, pred.shape[2])
            pred = pred.max(dim=2).values

        metrics = {}
        for cls in range(n_classes):
            for lb in range(n_labels):
                lb_key = f"{lb}/{ds_name}/{cls_names[cls]}"

                eer, eer_t = binary_eer(pred[:, cls, lb], gt[:, cls, lb]).values()
                if eer_t == 0.0:
                    # all negatives in batch
                    eer_t = (
                        trainer.callback_metrics[f"eer_t_{lb_key}"].item()
                        if f"eer_t_{lb_key}" in trainer.callback_metrics
                        else 0.5
                    )
                else:
                    # avoid skewing average with default values
                    metrics[f"eer_{lb_key}"] = eer
                    metrics[f"eer_t_{lb_key}"] = eer_t

                metrics[f"ap_{lb_key}"] = binary_average_precision(
                    pred[:, cls, lb], gt[:, cls, lb].long()
                )

                # calculate remaining metrics using equilibrium threshold
                metrics[f"recall_{lb_key}"] = binary_recall(
                    pred[:, cls, lb], gt[:, cls, lb], eer_t
                )
                metrics[f"precision_{lb_key}"] = binary_precision(
                    pred[:, cls, lb], gt[:, cls, lb], eer_t
                )
                metrics[f"f1_{lb_key}"] = binary_f1_score(
                    pred[:, cls, lb], gt[:, cls, lb], eer_t
                )

        pl_module.log_dict(metrics)
