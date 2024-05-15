import json
from typing import Sequence
from pathlib import Path
from abc import ABC

from common.label import Label


class Benchmark(ABC):
    data: Sequence[dict]  # Each element is of the form {"content", "label"}
    label_meaning: dict[str, Label]
    file_path: Path

    def __init__(self, name: str):
        self.name = name

    def get_labels(self):
        labels = []
        for instance in self:
            labels.append(instance["label"])
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AVeriTeC(Benchmark):
    label_meaning = {
        "Supported": Label.SUPPORTED,
        "Not Enough Evidence": Label.NEI,
        "Refuted": Label.REFUTED,
        "Conflicting Evidence/Cherrypicking": Label.CONFLICTING,
    }

    def __init__(self, variant="dev"):
        super().__init__("averitec")
        self.file_path = Path(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/AVeriTeC/{variant}.json")

        # Load the data
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        self.data = [{"content": d["claim"], "label": self.label_meaning[d["label"]]} for d in data]


class FEVER(Benchmark):
    data_labels_to_model_labels = {
        "SUPPORTS": Label.SUPPORTED,
        "NOT ENOUGH INFO": Label.NEI,
        "REFUTES": Label.REFUTED,
    }

    def __init__(self, variant="dev"):
        super().__init__("averitec")
        self.file_path = Path(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/FEVER/{variant}.json")

        # Load the data
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        self.data = [{"content": d["claim"], "label": self.data_labels_to_model_labels[d["label"]]} for d in data]
