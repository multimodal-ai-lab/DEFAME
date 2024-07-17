import random
from abc import ABC
from pathlib import Path
from typing import MutableSequence, Iterable

from src.common.action import Action
from src.common.label import Label
from .averitec.benchmark import AVeriTeC
from .fever.benchmark import FEVER
from .verite.benchmark import VERITE


class Benchmark(ABC, Iterable):
    shorthand: str

    data: MutableSequence[dict]  # Each element is of the form {"id": ..., "content": ..., "label": ...}

    is_multimodal: bool

    class_mapping: dict[str, Label]  # Maps the benchmark-specific class/label to the standard Label class
    class_definitions: dict[Label, str]  # Explains (to the LLM) the meaning of each class/label

    file_path: Path

    available_actions: list[Action]  # If none, all actions are allowed

    extra_prepare_rules: str = None  # Additional, benchmark-specific instructions to guide LLM's initial reasoning
    extra_plan_rules: str = None  # Additional, benchmark-specific instructions to guide LLM's action planning
    extra_judge_rules: str = None  # Additional, benchmark-specific instructions to guide LLM's verdict prediction

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
        return list(self.class_definitions.keys())

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


BENCHMARK_REGISTRY = {
    AVeriTeC,
    FEVER,
    VERITE
}


def load_benchmark(name: str, **kwargs) -> Benchmark:
    for benchmark in BENCHMARK_REGISTRY:
        if name == benchmark.shorthand:
            return benchmark(**kwargs)
    raise ValueError(f"No benchmark named '{name}'.")
