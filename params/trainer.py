from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from lib.callbacks.metrics import MetricsCallback
from lib.callbacks.wandb_reporting import WandbSamplesCallback
from params.datamodule import project


# to enable WandB
# wandb_logger = WandbLogger(
#     project=project,
#     log_model=True,
#     # save_dir="s3://path/to/checkpoints", # local or s3 path for checkpointing
# )

# Any logger from here will work: https://lightning.ai/docs/pytorch/stable/extensions/logging.html
# default - tensorboard

trainer = Trainer(
    # logger=wandb_logger, # to enable WandB
    log_every_n_steps=50,
    limit_train_batches=2000,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    max_epochs=2,
    callbacks=[
        ModelCheckpoint(
            save_last=True,
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            verbose=True,
            dirpath=None,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        MetricsCallback(),
        # WandbSamplesCallback(),
    ],
)
