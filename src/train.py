import json
import subprocess
import sys
from pathlib import Path
from pprint import pprint
import os
import argparse

# os.environ['WANDB_MODE'] = "disabled"
sys.path.append(Path(__file__).parent.parent.as_posix())

from params.model import model, ckpt_path
from params.datamodule import datamodule
from params.trainer import trainer #, wandb_logger

def main() -> None:
    # wandb_logger.watch(model, log="all")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    # wandb_logger.experiment.unwatch(model)
    valid_metrics = trainer.validate(model, datamodule=datamodule)
    pprint(valid_metrics)


if __name__ == "__main__":
    main()



