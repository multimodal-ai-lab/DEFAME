import json
from typing import Sequence
from pathlib import Path
from abc import ABC


class Benchmark(ABC):
    data: Sequence[dict]  # Each element is of the form {"content", "label"}
    labels: set[str]
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
    labels = {"Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"}

    def __init__(self, variant="dev"):
        super().__init__("averitec")
        self.file_path = Path(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/AVeriTeC/{variant}.json")

        # Load the data
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        self.data = [{"content": d["claim"], "label": d["label"]} for d in data]


class FEVER(Benchmark):
    labels = {"SUPPORTS", "NOT ENOUGH INFO", "REFUTES"}

    def __init__(self, variant="dev"):
        super().__init__("averitec")
        self.file_path = Path(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/FEVER/{variant}.json")

        # Load the data
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        self.data = [{"content": d["claim"], "label": d["label"]} for d in data]
