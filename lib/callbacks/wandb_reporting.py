from collections import defaultdict
from typing import Optional, Any, List

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import wandb
import numpy as np
import torch


class WandbSamplesCallback(Callback):
    def __init__(
        self,
        n_samples: int = 8,
        log_interval_epochs: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.log_interval_epochs = log_interval_epochs

        self.gt_labels = defaultdict(list)
        self.pred_labels = defaultdict(list)

        self.sample_from_batch = None

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

        gt_labels = batch["multilabel"].cpu()
        gt_scores = batch["score"].cpu()

        pred_labels = torch.sigmoid(outputs["logits"]["multilabel"]).cpu()
        if not separate_sides and pred_labels.shape[1] == n_classes * 2:
            # merge left and right AU when train is on separate and val is on joined sides
            pred_labels = pred_labels.view(
                pred_labels.shape[0], -1, 2, pred_labels.shape[2]
            )
            pred_labels = pred_labels.max(dim=2).values

        # acquire thresholds for each logit
        eer_t = torch.ones([n_classes, n_labels]) * 0.5
        for cls in range(n_classes):
            for lb in range(n_labels):
                key = f"eer_t_{lb}/{ds_name}/{cls_names[cls]}"
                if key in trainer.callback_metrics:
                    eer_t[cls, lb] = trainer.callback_metrics[key]

        pred_scores = torch.sum((pred_labels > eer_t).float(), dim=1) / n_labels

        self.gt_labels[dataloader_idx].extend(gt_labels)
        self.pred_labels[dataloader_idx].extend(pred_labels)

        if (
            not batch_idx
            == (
                self.sample_from_batch[dataloader_idx]
                if self.sample_from_batch is not None
                else 0
            )
            or not pl_module.current_epoch % self.log_interval_epochs == 0
        ):
            return

        wandb_logger = pl_module.logger
        batch_size = batch["img"].shape[0]
        indices = np.random.choice(
            list(range(batch_size)), size=self.n_samples, replace=False
        )

        imgs = [img for img in batch["img"][indices]]

        gt_scores = gt_scores[indices]
        pred_scores = pred_scores[indices]

        columns = ["image", "gt_scores", "pred_scores", "step"]
        data = []
        for img, gt_s, pred_s in zip(imgs, gt_scores, pred_scores):
            data.append(
                [
                    wandb.Image(img),
                    gt_s,
                    pred_s,
                    trainer.global_step,
                ]
            )

        wandb_logger.log_table(
            key=f"val_samples/dataloader_idx_{dataloader_idx}",
            columns=columns,
            data=data,
        )

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        cm = {}
        prc = {}

        for dataloader_idx in self.gt_labels.keys():
            n_labels = pl_module.heads["multilabel"].logits_per_class
            ds_name = trainer.val_dataloaders[dataloader_idx].dataset.name
            cls_names = trainer.val_dataloaders[dataloader_idx].dataset.cls_names
            n_classes = len(cls_names)

            gt = torch.stack(self.gt_labels[dataloader_idx])
            pred = torch.stack(self.pred_labels[dataloader_idx])

            # shape = n_samples x n_classes x n_labels

            for cls in range(n_classes):
                for lb in range(n_labels):
                    p = pred[:, cls, lb].detach().cpu().numpy()
                    g = gt[:, cls, lb].detach().cpu().numpy()
                    lb_key = f"{lb}/{ds_name}/{cls_names[cls]}"

                    eer_t = (
                        trainer.callback_metrics[f"eer_t_{lb_key}"].detach().cpu().numpy()
                        if f"eer_t_{lb_key}" in trainer.callback_metrics
                        else 0.5
                    )

                    cm[f"cm_{lb_key}"] = wandb.plot.confusion_matrix(
                        y_true=(g > eer_t).astype(int),
                        preds=(p > eer_t).astype(int),
                        class_names=["neg", "pos"],
                    )

                    p_split = np.vstack((1 - p, p)).transpose()
                    prc[f"prc/{lb_key}"] = wandb.plot.pr_curve(
                        y_true=(g > eer_t).astype(int),
                        y_probas=p_split,
                        labels=["neg", "pos"],
                    )

        wandb.log(cm | prc)

        self.gt_labels.clear()
        self.pred_labels.clear()

        self.sample_from_batch = np.random.randint(low=0, high=trainer.num_val_batches)
