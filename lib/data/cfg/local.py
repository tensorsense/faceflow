from dataclasses import dataclass, field
from typing import List
from pathlib import Path

@dataclass
class LocalNaturalDatasetCfg:
    name: str
    root: str
    labels_filename: str = "au.csv"
    crops_dir: str = "crops"
    aus: List[str] = field(
        default_factory=lambda: [
            "AU1",
            "AU2",
            "AU4",
        ]
    )

# TODO: LocalSyntheticDatasetCfg