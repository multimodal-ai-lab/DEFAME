import json
from abc import ABC
from pathlib import Path
from typing import MutableSequence, Iterable, Iterator
import random

import orjsonl

from common.label import Label
from common.shared_config import path_to_data


class Benchmark(ABC, Iterable):
    data: MutableSequence[dict]  # Each element is of the form {"id": ..., "content": ..., "label": ...}
    label_mapping: dict[str, Label]
    file_path: Path

    def __init__(self, name: str):
        self.name = name

    def get_labels(self) -> list[Label]:
        """Returns the ground truth labels of this dataset as a list."""
        labels = []
        for instance in self:
            labels.append(instance["label"])
        return labels

    def get_classes(self) -> list[Label]:
        """Returns a list of distinct labels representing the classes occurring in this dataset."""
        return list(self.label_mapping.values())

    def shuffle(self):
        """Reorders the samples randomly."""
        random.shuffle(self.data)

    def get_by_id(self, claim_id: int):
        """Returns the instance with the given ID (different from the instance's index)."""
        for instance in self:
            if instance["id"] == claim_id:
                return instance
        raise ValueError(f"No instance with ID {claim_id} was found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for instance in self.data:
            yield instance


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

        self.data = [{"id": i,
                      "content": d["claim"],
                      "label": self.label_mapping[d["label"]]}
                     for i, d in enumerate(data)]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


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

        self.data = [{"id": i,
                      "content": d["claim"],
                      "label": self.label_mapping[d["label"].lower()]}
                     for i, d in enumerate(data)]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


def load_benchmark(name: str, **kwargs) -> Benchmark:
    match name:
        case "fever":
            return FEVER(**kwargs)
        case "averitec":
            return AVeriTeC(**kwargs)
