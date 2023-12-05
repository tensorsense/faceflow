from functools import partial
from typing import Dict, List

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
import lightning.pytorch as pl
from timm.scheduler.scheduler import Scheduler


class AUHead(torch.nn.Module):
    def __init__(
        self,
        task: str,
        in_channels: int,
        num_classes: int = 1,
        logits_per_class: int = 1,
        act: torch.nn.Module = torch.nn.Identity(),
        loss: torch.nn.Module = None,
        weight: float = 1.0,
    ):
        """
        A module that wraps task key, prediction head, corresponding activation, loss and loss weight
        :param task: key for the task name, must match the gt key from the datamodule
        :param in_channels: timm feature extractor's feature_info.channels()[-1]
        :param num_classes: must match the gt
        :param logits_per_class: number of outputs for each class; use > 1 for multilabel classification
        :param act: activation to be used on logits during prediction
        :param loss: loss function to be used during training and validation to evaluate head's output
        :param weight: loss weight
        """
        super().__init__()

        assert task in {"multilabel"}
        self.task = task
        self.loss = loss
        self.num_classes = num_classes
        self.weight = weight
        self.logits_per_class = logits_per_class

        self.head = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes * logits_per_class,
                kernel_size=1,
            ),
            nn.AdaptiveAvgPool2d(1),
        )
        self.act = torch.nn.Identity() if act is None else act

    def forward(self, x):
        x = self.head(x)
        return x.view(-1, self.num_classes, self.logits_per_class)

    def predict(self, x):
        x = self(x)
        x = self.act(x)
        return x


class AUModel(pl.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        heads_partials: List[partial[AUHead]],
        optimizer_partial: partial = None,
        scheduler_partial: partial = None,
        test_metrics: Dict[str, partial] = None,
    ):
        """
        Lightning wrapper that encapsulates training and validation workflows.
        :param backbone: timm-compatible feature extractor
        :param heads_partials: partials of AUHead that will be completed with backbone out channels
        :param optimizer_partial: partial of an optimizer that will get model parameters passed into it
        :param scheduler_partial: partial of a scheduled that will get an optimizer passed into it
        :param test_metrics: dict of functional metrics to be used during test phase
        """
        super().__init__()
        self.backbone = backbone
        self.head_partials = heads_partials
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial
        self.test_metrics = test_metrics

        _heads = [
            h(in_channels=self.backbone.feature_info.channels()[-1])
            for h in self.head_partials
        ]
        self.heads: torch.nn.ModuleDict[str, AUHead] = torch.nn.ModuleDict(
            {h.task: h for h in _heads}
        )

        print(f"Backbone reduction: {self.backbone.feature_info.reduction()}")
        print(f"Backbone channels: {self.backbone.feature_info.channels()}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.backbone(x)[-1]
        outs = {task: head(x).squeeze() for task, head in self.heads.items()}
        return outs

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.backbone(x)[-1]
        outs = {task: head.predict(x).squeeze() for task, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image = batch["img"]
        x = self(image)

        assert set(x.keys()).issubset(
            batch.keys()
        ), f"Missing gt for pred keys: gt {batch.keys()}, pred {x.keys()}"

        loss = torch.zeros(1, device=self.device)
        for task, head in self.heads.items():
            loss += head.loss(x[task], batch[task]) * head.weight

        self.log("train_loss", loss)
        return {"loss": loss, "logits": x}

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        image = batch["img"]
        x = self(image)

        assert set(x.keys()).issubset(
            batch.keys()
        ), f"Missing gt for pred keys: gt {batch.keys()}, pred {x.keys()}"

        loss = torch.zeros(1, device=self.device)
        for task, head in self.heads.items():
            pred = x[task]
            if batch[task].shape[1] == pred.shape[1] // 2:
                # Assuming joined sides, merging predictions..."
                pred = pred.view(pred.shape[0], -1, 2, pred.shape[2])
                pred = pred.max(dim=2).values

            loss += head.loss(pred, batch[task]) * head.weight

        self.log("val_loss", loss)
        return {"loss": loss, "logits": x}

    def configure_optimizers(self):
        assert (
            self.optimizer_partial is not None
        ), "Optimizer was not provided on initialization"
        assert (
            self.scheduler_partial is not None
        ), "Scheduler was not provided on initialization"

        optimizer = self.optimizer_partial(self.parameters())
        scheduler = self.scheduler_partial(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: Scheduler, metric):
        scheduler.step(
            epoch=self.current_epoch
        )  # timm's scheduler needs the epoch value
