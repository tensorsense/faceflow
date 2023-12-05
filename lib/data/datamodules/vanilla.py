from typing import Dict, List

import albumentations as A
import lightning.pytorch as pl
from torch.utils.data import DataLoader, ConcatDataset

from lib.data.datasets.vanilla import SimpleAUDataset


class AUDataModule(pl.LightningDataModule):

    def __init__(self, dataset_cfg: Dict[str, List],
                 image_size: int = 224,
                 logits_per_class: int = 2,
                 train_transforms: A.Compose = None,
                 val_transforms: A.Compose = None,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 random_state: int = 1337):
        """
        Wrapper that abstracts away data handling, like instantiating datasets, setting dataloaders etc.
        :param dataset_cfg: dict with {'train', 'val'} keys, each item contains a list of dict configs for datasets used
        during the corresponding stage. The config has to include name, root, images and labels paths.
        :param image_size: to what size the input is going to be rescaled and cropped
        :param train_transforms:
        :param val_transforms:
        :param batch_size:
        :param num_workers:
        :param random_state:
        """
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.image_size = image_size
        self.logits_per_class = logits_per_class

        self.train_transform = train_transforms
        self.val_transform = val_transforms

        self.train_dataset = None
        self.val_datasets = None

        self.random_state = random_state
        self.num_workers = num_workers
        self.batch_size = batch_size

    def fetch_datasets(self, mode="train"):
        assert mode in {"train", "val"}
        cfg = self.dataset_cfg[mode]

        datasets = []
        for ds in cfg:
            datasets.append(SimpleAUDataset(name=ds.name,
                                            root=ds.root,
                                            crops_dir=ds.crops_dir,
                                            labels_filename=ds.labels_filename,
                                            aus=ds.aus,
                                            transform=self.train_transform if mode in {"train"} else self.val_transform,
                                            logits_per_class=self.logits_per_class))
        return datasets

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = ConcatDataset(self.fetch_datasets("train"))
            self.val_datasets = self.fetch_datasets("val")

            print(f"Train size: {len(self.train_dataset)}")
            print(f"Val sizes: {[len(d) for d in self.val_datasets]}")

        if stage == "test":
            self.val_datasets = self.fetch_datasets("val")

        if stage == "predict":
            self.val_datasets = self.fetch_datasets("val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return [DataLoader(d, batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers) for d in self.val_datasets]

    def test_dataloader(self):
        return [DataLoader(d, batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers) for d in self.val_datasets]

    def predict_dataloader(self):
        return [DataLoader(d, batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers) for d in self.val_datasets]
