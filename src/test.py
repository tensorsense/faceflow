import sys
from functools import partial
from pathlib import Path
import torchmetrics

sys.path.append(Path(__file__).parent.parent.as_posix())

from params.model import model, ckpt_path
from params.datamodule import datamodule
from params.trainer import trainer


def main() -> None:

    test_metrics = {
        "mae": partial(torchmetrics.functional.mean_absolute_error),
        "rmse": partial(torchmetrics.functional.mean_squared_error, squared=False),
        "mape": partial(torchmetrics.functional.mean_absolute_percentage_error),
        "smape": partial(torchmetrics.functional.symmetric_mean_absolute_percentage_error),
        "wmape": partial(torchmetrics.functional.weighted_mean_absolute_percentage_error),
    }

    model.test_metrics = test_metrics
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()



