import json
import orjsonl
from typing import Sequence, Iterable
from pathlib import Path
from abc import ABC
from common.shared_config import path_to_data

from common.label import Label


class Benchmark(ABC, Iterable):
    data: Sequence[dict]  # Each element is of the form {"content", "label"}
    label_mapping: dict[str, Label]
    file_path: Path

    def __init__(self, name: str):
        self.name = name

    def get_labels(self) -> list[Label]:
        labels = []
        for instance in self:
            labels.append(instance["label"])
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_classes(self):
        return list(self.label_mapping.values())


class AVeriTeC(Benchmark):
    label_mapping = {
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

        self.data = [{"content": d["claim"], "label": self.label_mapping[d["label"]]} for d in data]


class FEVER(Benchmark):
    label_mapping = {
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
                      "label": self.label_mapping[d["label"].lower()]}
                     for d in data]


def load_benchmark(name: str, **kwargs) -> Benchmark:
    match name:
        case "fever": return FEVER(**kwargs)
        case "averitec": return AVeriTeC(**kwargs)
