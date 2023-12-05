from pathlib import Path
from typing import Dict, List

import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader, ConcatDataset

import lib.data.model.local
from lib.data.datasets.ssl import LabeledDataset, UnlabeledDataset


class SSLDataModule(pl.LightningDataModule):
    def __init__(self, dataset_cfg: Dict[str, List[lib.data.model.local.Dataset]],
                 image_size: int = 224,
                 logits_per_class: int = 2,
                 strong_transforms: A.Compose = None,
                 weak_transforms: A.Compose = None,
                 val_transforms: A.Compose = None,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 random_state: int = 1337):
        """

        :param dataset_cfg:
        :param image_size:
        :param logits_per_class:
        :param strong_transforms:
        :param weak_transforms:
        :param val_transforms:
        :param batch_size:
        :param num_workers:
        :param random_state:
        """
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.image_size = image_size
        self.logits_per_class = logits_per_class

        self.strong_transform = strong_transforms
        self.weak_transform = weak_transforms
        self.val_transform = val_transforms

        self.train_ds_labeled = None
        self.train_ds_unlabeled = None
        self.val_datasets = None

        self.random_state = random_state
        self.num_workers = num_workers
        self.batch_size = batch_size

    def fetch_labeled_datasets(self, mode="train_labeled"):
        assert mode in {"train_labeled", "val"}
        cfg = self.dataset_cfg[mode]

        datasets = []
        for ds in cfg:
            datasets.append(LabeledDataset(name=ds.name,
                                           root=ds.root,
                                           images_dir=ds.images_dir,
                                           labels_filename=ds.labels_filename,
                                           weak_transform=self.val_transform if mode in {
                                               "val"} else self.weak_transform,
                                           strong_transform=self.strong_transform,
                                           logits_per_class=self.logits_per_class))
        return datasets

    def fetch_unlabeled_datasets(self, mode):
        assert mode in {"train_unlabeled"}
        cfg = self.dataset_cfg[mode]

        datasets = []
        for ds in cfg:
            datasets.append(UnlabeledDataset(name=ds.name,
                                             root=ds.root,
                                             images_dir=ds.images_dir,
                                             strong_transform=self.strong_transform,
                                             weak_transform=self.weak_transform))
        return datasets

    def setup(self, stage: str):
        if stage == "fit":
            labeled_datasets = self.fetch_labeled_datasets("train_labeled")
            unlabeled_datasets = self.fetch_unlabeled_datasets(
                "train_unlabeled") if "train_unlabeled" in self.dataset_cfg else []

            self.train_ds_labeled = ConcatDataset(labeled_datasets)
            self.train_ds_unlabeled = ConcatDataset(unlabeled_datasets) if len(unlabeled_datasets) > 0 else None
            self.val_datasets = self.fetch_labeled_datasets("val")

            print(f"Train size: "
                  f"labeled {len(self.train_ds_labeled)}, "
                  f"unlabeled {len(self.train_ds_unlabeled) if self.train_ds_unlabeled else 0}")
            print(f"Val sizes: {[len(d) for d in self.val_datasets]}")

        if stage == "test":
            self.val_datasets = self.fetch_labeled_datasets("val")

        if stage == "predict":
            self.val_datasets = self.fetch_labeled_datasets("val")

    def train_dataloader(self):
        loaders = {"labeled": DataLoader(self.train_ds_labeled,
                                         batch_size=self.batch_size - self.batch_size // 2
                                         if self.train_ds_unlabeled is not None else self.batch_size,
                                         shuffle=True,
                                         num_workers=self.num_workers)}
        if self.train_ds_unlabeled is not None:
            loaders["unlabeled"] = DataLoader(self.train_ds_unlabeled,
                                              batch_size=self.batch_size // 2,
                                              shuffle=True,
                                              num_workers=self.num_workers)

        return CombinedLoader(loaders, mode="min_size")

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
