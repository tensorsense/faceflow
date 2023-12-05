from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch

from lib.data.datasets.vanilla import SimpleAUDataset


class LabeledDataset(SimpleAUDataset):
    def __init__(
        self,
        name: str,
        root: str,
        labels_filename: str,
        aus: List[str],
        logits_per_class: int = 2,
        strong_transform=None,
        weak_transform=None,
    ):
        super().__init__(
            name=name,
            root=root,
            labels_filename=labels_filename,
            aus=aus,
            logits_per_class=logits_per_class,
        )
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform

    def __getitem__(self, idx):
        info = self.info_df.iloc[idx]

        img = cv2.imread(info.filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        img_w = self.weak_transform(image=img)["image"]
        img_s = (
            self.strong_transform(image=img)["image"]
            if self.strong_transform is not None
            else None
        )

        scores = info[:-1].copy()  # skip the filename
        labels = scores.apply(self.encode_score)
        labels = np.vstack(labels.values)

        scores = torch.tensor(scores)
        labels = torch.from_numpy(labels).float()

        return {"img": img_w, "multilabel": labels, "score": scores} | (
            {"img_s": img_s} if img_s is not None else {}
        )


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name: str,
        root: str,
        images_dir: str,
        exclude_filename: str = None,
        strong_transform=None,
        weak_transform=None,
    ):
        self.name = name
        self.root = Path(root)
        self.images_path = self.root.joinpath(images_dir)
        self.exclude_path = (
            self.root.joinpath(exclude_filename)
            if exclude_filename is not None
            else None
        )
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform

        self.exclude_df = (
            pd.read_csv(self.exclude_path)
            if self.exclude_path is not None
            else pd.DataFrame(columns=["face_id"])
        )
        self.img_path_list = [
            p
            for p in self.images_path.glob("*")
            if p not in self.exclude_df.face_id.tolist()
        ]

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]

        img = cv2.imread(img_path.as_posix(), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        img_ulb_s = self.strong_transform(image=img)["image"]
        img_ulb_w = self.weak_transform(image=img)["image"]

        return {"img_ulb_s": img_ulb_s, "img_ulb_w": img_ulb_w}
