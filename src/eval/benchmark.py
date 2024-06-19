import json
import random
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import MutableSequence, Iterable, Iterator

import orjsonl

from common.claim import Claim
from common.label import Label
from common.shared_config import path_to_data


class Benchmark(ABC, Iterable):
    data: MutableSequence[dict]  # Each element is of the form {"id": ..., "content": ..., "label": ...}
    class_mapping: dict[str, Label]  # Maps the benchmark-specific class/label to the standard Label class
    class_definitions: dict[Label, str]  # Explains (to the LLM) the meaning of each class/label
    file_path: Path
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


class AVeriTeC(Benchmark):
    class_mapping = {
        "Supported": Label.SUPPORTED,
        "Not Enough Evidence": Label.NEI,
        "Refuted": Label.REFUTED,
        "Conflicting Evidence/Cherrypicking": Label.CONFLICTING,
    }

    class_definitions = {
        Label.SUPPORTED: "The knowledge from the fact-check supports or at least strongly implies the Claim. "
                         "Mere plausibility is not enough for this decision.",
        Label.NEI: "The fact-check does not contain sufficient information to come to a conclusion. For example, "
                   "there is substantial lack of evidence. In this case, state which information exactly "
                   "is missing. In particular, if no RESULTS or sources are available, pick this decision.",
        Label.REFUTED: "The knowledge from the fact-check explicitly and clearly refutes the Claim. The mere "
                       "absence or lack of supporting evidence is not enough for being refuted (argument "
                       "from ignorance).",
        Label.CONFLICTING: "The knowledge from the fact-check contains conflicting evidence from multiple "
                           "reliable, up-to-date, non-refuted sources, even after extensive fact-checking research.",
        Label.CHERRY_PICKING: "The Claim is supported or refuted, however it ignores important facts that, "
                              "when added to the Claim, create a significantly different impression. Pick this "
                              "decision also in the case if the Claim is not universally true but true under "
                              "certain conditions.",
    }

    def __init__(self, variant="dev"):
        super().__init__(f"averitec_{variant}")
        self.file_path = Path(path_to_data + f"AVeriTeC/{variant}.json")

        # Load the data
        with open(self.file_path, 'r') as f:
            data = json.load(f)

        self.data = [{"id": i,
                      "content": Claim(
                          text=d["claim"],
                          author=d["speaker"],
                          date=datetime.strptime(d["claim_date"], "%d-%m-%Y"),
                          origin=d["original_claim_url"]
                      ),
                      "label": self.class_mapping[d["label"]]}
                     for i, d in enumerate(data)]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


class FEVER(Benchmark):
    class_mapping = {
        "supports": Label.SUPPORTED,
        "not enough info": Label.NEI,
        "refutes": Label.REFUTED,
    }

    class_definitions = {
        Label.SUPPORTED:
            """The knowledge from the fact-check explicitly supports the entire Claim.
            That is, there is at least (!) one source that clearly (directly or
            indirectly) implies the Claim. Mere plausibility or the absence of
            opposing evidence is not enough for this decision.""",
        Label.REFUTED:
            """The knowledge from the fact-check explicitly refutes (at leas a part of) the Claim.
            That is, there is at least (!) one source that clearly (directly or
            indirectly) opposes the Claim. Mere plausibility or the absence of
            supporting evidence is not enough for this decision.""",
        Label.NEI:
            """Pick this decision if the Claim is neither `supported` nor `refuted`. For
            example, pick this decision if there is still evidence needed to clearly verify
            or refute the Claim. Before picking this decision, state which information exactly
            is missing."""
    }

    extra_prepare_rules = """* Before you start, begin with a _grammar check_ of the Claim. If it
    has some grammatical errors, there is a high chance that the Claim means something different
    than understandable at first glance. Take grammatical errors serious and elaborate on them.
    * **Take the Claim literally**: Assume that each word of the Claim is as intended. Be strict
    with the interpretation of the Claim.
    * The Claim stems from a fact-checking challenge. A human fabricated the Claim artificially 
    by using Wikipedia. The Claim could be a misleading prank, just like a trick question. It may also require
    a chain of multiple investigation steps, re-using previously retrieved knowledge."""
    extra_plan_rules = """* The Claim stems from a fact-checking challenge. A human engineered the Claim
    artificially by using Wikipedia. The Claim could be misleading, just like a trick question. It may
    also require a chain of multiple investigation steps, re-using previously retrieved knowledge."""

    def __init__(self, variant="dev"):
        super().__init__(f"fever_{variant}")
        self.file_path = Path(path_to_data + f"FEVER/{variant}.jsonl")

        # Load the data
        data = orjsonl.load(self.file_path)

        self.data = [{"id": i,
                      "content": Claim(d["claim"]),
                      "label": self.class_mapping[d["label"].lower()]}
                     for i, d in enumerate(data)]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


def load_benchmark(name: str, **kwargs) -> Benchmark:
    match name:
        case "fever":
            return FEVER(**kwargs)
        case "averitec":
            return AVeriTeC(**kwargs)
