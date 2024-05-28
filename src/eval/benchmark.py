import json
import orjsonl
from typing import Sequence
from pathlib import Path
from abc import ABC
from common.shared_config import path_to_data

from common.label import Label


class Benchmark(ABC):
    data: Sequence[dict]  # Each element is of the form {"content", "label"}
    label_meaning: dict[str, Label]
    file_path: Path

    def __init__(self, name: str):
        self.name = name

    def get_labels(self) -> list[str]:
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
        super().__init__(f"averitec_{variant}")
        self.file_path = Path(path_to_data + f"AVeriTeC/{variant}.json")

        # Load the data
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        self.data = [{"content": d["claim"], "label": self.label_meaning[d["label"]]} for d in data]


class FEVER(Benchmark):
    data_labels_to_model_labels = {
        "supports": Label.SUPPORTED,
        "not enough info": Label.NEI,
        "refutes": Label.REFUTED,
    }

    def __init__(self, variant="dev"):
        super().__init__(f"fever_{variant}")
        self.file_path = Path(path_to_data + f"FEVER/{variant}.jsonl")

        # Load the data
        data = orjsonl.load(self.file_path)

        self.data = [{"content": d["claim"],
                      "label": self.data_labels_to_model_labels[d["label"].lower()]}
                     for d in data]


def load_benchmark(name: str, **kwargs):
    match name:
        case "fever": return FEVER(**kwargs)
        case "averitec": return AVeriTeC(**kwargs)
