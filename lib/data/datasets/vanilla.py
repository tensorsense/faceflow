from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch


class SimpleAUDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name: str,
        root: str,
        crops_dir: str,
        labels_filename: str,
        aus: List[str],
        transform=None,
        logits_per_class: int = 2,
    ):
        """
        Dataset generated using OpenFace action unit recognition
        :param root: directory that encloses the dataset
        :param labels_filename:
        :param aus:
        :param transform: Albumentations transforms to apply to the data
        :param logits_per_class:
        """
        self.name = name
        self.root = Path(root)
        self.crops_dir = Path(crops_dir) if crops_dir else self.root
        self.labels_path = self.root.joinpath(labels_filename)
        self.aus = aus
        self.transform = transform
        self.logits_per_class = logits_per_class

        self.info_df = self.load_info()

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        info = self.info_df.iloc[idx]

        img = cv2.imread((self.crops_dir / info.filename).as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]

        scores = info[:-1].copy()  # skip the filename, it was set as last
        labels = scores.apply(self.encode_score)
        labels = np.vstack(labels.values)

        scores = torch.tensor(scores)
        labels = torch.from_numpy(labels).float()

        return {"img": img, "multilabel": labels, "score": scores}

    def load_info(self):
        labels_df = pd.read_csv(self.labels_path)
        labels_df = labels_df[self.aus + ['filename']]  # filter out extra columns, filename is set last
        return labels_df

    def encode_score(self, score):
        if self.logits_per_class == 1:
            label = np.array(score > 0.0, dtype=np.float32)
        else:
            thresholds = np.linspace(0, 1, self.logits_per_class + 1, endpoint=False)[
                1:
            ]
            label = (thresholds < score).astype(np.float32)
        return label
