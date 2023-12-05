from functools import partial

import timm
import torch
from timm.scheduler import CosineLRScheduler
from torch.nn import BCEWithLogitsLoss
from torchvision.ops.focal_loss import sigmoid_focal_loss

from lib import core
from lib.core import AUModel
from lib.losses.db import DistributionBalancedLoss

from params.datamodule import datamodule, logits_per_class, num_aus


backbone = timm.create_model(
    "convnextv2_nano.fcmae_ft_in22k_in1k", pretrained=True, features_only=True
)

# loss = BCEWithLogitsLoss()

loss = partial(sigmoid_focal_loss, reduction="mean")

# datamodule.setup("fit")
# train_datasets = datamodule.train_dataset.datasets
# pos_counts = sum([ds.pos_counts for ds in train_datasets])
# neg_counts = sum([ds.neg_counts for ds in train_datasets])
# loss = DistributionBalancedLoss(pos_counts=pos_counts, neg_counts=neg_counts)


# def binary_dice_loss(inputs, targets, smooth=1e-4):
#     inputs = torch.nn.functional.sigmoid(inputs.squeeze())
#     inputs = inputs.view(-1)
#     targets = targets.view(-1)
#
#     intersection = (inputs * targets).sum()
#     dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
#     return 1 - dice
#
#
# loss = binary_dice_loss

heads_partials = [
    partial(
        core.AUHead,
        task="multilabel",
        num_classes=num_aus,
        logits_per_class=logits_per_class,
        loss=loss,
    )
]

optimizer_partial = partial(torch.optim.Adam, weight_decay=0.00001, lr=0.00001)

scheduler_partial = partial(
    CosineLRScheduler,
    t_initial=10,
    lr_min=0.000001,
    cycle_decay=0.5,
    warmup_t=2,
    warmup_lr_init=0.000001,
    warmup_prefix=True,
    cycle_limit=5,
    cycle_mul=2,
)

model = AUModel(
    backbone=backbone,
    heads_partials=heads_partials,
    optimizer_partial=optimizer_partial,
    scheduler_partial=scheduler_partial,
)

ckpt_path = None
