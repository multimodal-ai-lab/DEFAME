import json
import os
import random
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import MutableSequence, Iterable, Iterator

import orjsonl
import pandas as pd
from PIL import Image

from config.globals import data_base_dir
from infact.common.action import Action, WebSearch, WikiDumpLookup
from infact.common.content import Content
from infact.common.label import Label


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

    def __init__(self, name: str, variant: str):
        self.name = name
        self.variant = variant  # 'train', 'dev' or 'test'

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

    def get_class_name(self, label: Label) -> str:
        """Returns the original class name for the given standard Label."""
        for name, cls in self.class_mapping.items():
            if cls == label:
                return name
        raise ValueError(f"Unknown label {label}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for instance in self.data:
            yield instance


class AVeriTeC(Benchmark):
    shorthand = "averitec"

    is_multimodal = False

    class_mapping = {
        "Supported": Label.SUPPORTED,
        "Not Enough Evidence": Label.NEI,
        "Refuted": Label.REFUTED,
        "Conflicting Evidence/Cherrypicking": Label.CONFLICTING,
    }

    class_definitions = {
        Label.SUPPORTED:
            """The knowledge from the fact-check supports or at least strongly implies the Claim.
            Mere plausibility is not enough for this decision.""",
        Label.NEI:
            """The fact-check does not contain sufficient information to come to a conclusion. In particular,
            there is substantial lack of both supporting and refuting evidence.""",
        Label.REFUTED:
            """The knowledge from the fact-check explicitly and clearly refutes at least substantial parts
            if not even the whole the Claim.""",
        Label.CONFLICTING:
            """The Claim has both supporting and refuting evidence from multiple sources.""",
        Label.CHERRY_PICKING:
            """The Claim is technically true but misleads by excluding important context. Including
            that context would create a significantly different impression. Pick this
            decision also in the case if the Claim is not universally true but true under
            certain conditions.""",
    }

    extra_judge_rules = """* **Do not commit the "argument from ignorance" fallacy**: The absence of evidence
    for the Claim does NOT prove that the Claim is refuted. Instead, the Claim is simply unsupported--which is a case of 
    'not enough information'."""

    available_actions = [WebSearch]

    def __init__(self, variant="dev"):
        super().__init__(f"AVeriTeC ({variant})", variant)
        self.file_path = Path(data_base_dir + f"AVeriTeC/{variant}.json")

        # Load the data
        with open(self.file_path, 'r') as f:
            data_raw = json.load(f)

        data = []
        for i, d in enumerate(data_raw):
            date = d["claim_date"]
            content = Content(
                text=d["claim"],
                author=d["speaker"],
                date=datetime.strptime(date, "%d-%m-%Y") if date else None,
                origin=d["original_claim_url"],
                id_number=i
            )
            label = self.class_mapping[d["label"]] if variant in ["train", "dev"] else None
            justification = d["justification"] if variant in ["train", "dev"] else None

            data.append({"id": i, "content": content, "label": label, "justification": justification})

        self.data = data

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


class FEVER(Benchmark):
    shorthand = "fever"

    is_multimodal = False

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

    extra_prepare_rules_v1 = """* **Identify the altered segments**: Since the Claim is generated by altering
    sentences from Wikipedia, pinpoint the parts of the Claim that seem modified or out of place.
    * **Consider potential misleading elements**: The Claim may contain misleading or confusing elements due to
    the alteration process.
    * **Prepare to investigate original context**: Since the Claim is derived from Wikipedia sentences, be prepared
    to trace back to the original context or related information for accurate verification."""

    extra_plan_rules_v1 = """* **Assume the Claim may be misleading**: Since the Claim is generated by altering
    Wikipedia sentences, consider that it might be intentionally misleading or designed to test the fact-checking
    process.
    * **Identify key elements**: Break down the Claim into its key components and identify which parts require
    verification.
    * **Plan for multiple investigation steps**: The Claim may require a series of verification steps, including
    checking the original Wikipedia context, cross-referencing related information, and verifying altered segments.
    * **Consider alternative interpretations**: Given the altered nature of the Claim, consider multiple
    interpretations and verify each to ensure thorough fact-checking.
    * **Reuse previously retrieved knowledge**: Be prepared to reuse information and evidence gathered during
    previous verification steps to form a comprehensive judgment."""

    extra_prepare_rules_v2 = """* Before you start, begin with a _grammar check_ of the Claim. If it
    has some grammatical errors, there is a high chance that the Claim means something different
    than understandable at first glance. Take grammatical errors serious and elaborate on them.
    * **Take the Claim literally**: Assume that each word of the Claim is as intended. Be strict
    with the interpretation of the Claim.
    * The Claim stems from a fact-checking challenge. A human fabricated the Claim artificially 
    by using Wikipedia. The Claim could be a misleading prank, just like a trick question. It may also require
    a chain of multiple investigation steps, re-using previously retrieved knowledge."""

    extra_plan_rules_v2 = """* The Claim stems from a fact-checking challenge. A human engineered the Claim
    artificially by using Wikipedia. The Claim could be misleading, just like a trick question. It may
    also require a chain of multiple investigation steps, re-using previously retrieved knowledge."""

    available_actions = [WikiDumpLookup]

    def __init__(self, version=1, variant="dev"):
        super().__init__(f"FEVER V{version} ({variant})", variant)
        self.file_path = Path(data_base_dir + f"FEVER/fever{version}_{variant}.jsonl")
        self.justifications_file_path = Path(data_base_dir + f"FEVER/gt_justification_fever{version}_{variant}.jsonl")

        self.data = self.load_data(variant)

        if version == 1:
            self.extra_prepare_rules = self.extra_prepare_rules_v1
            self.extra_plan_rules = self.extra_plan_rules_v1
        elif version == 2:
            self.extra_prepare_rules = self.extra_prepare_rules_v2
            self.extra_plan_rules = self.extra_plan_rules_v2
        else:
            raise ValueError(f"Invalid FEVER version '{version}' specified.")

    def load_data(self, variant) -> list[dict]:
        # Read the files
        raw_data = orjsonl.load(self.file_path)
        if os.path.exists(self.justifications_file_path):
            justifications = orjsonl.load(self.justifications_file_path)
        else:
            justifications = None

        # Translate raw data into structured list of dicts
        data = []
        for i, row in enumerate(raw_data):
            content = Content(row["claim"])
            label = self.class_mapping[row["label"].lower()] if variant in ["train", "dev"] else None
            justification = justifications[i] if justifications is not None else None
            data.append({"id": i,
                         "content": content,
                         "label": label,
                         "justification": justification})

        return data

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


class VERITE(Benchmark):
    shorthand = "verite"

    is_multimodal = True

    class_mapping = {
        "true": Label.SUPPORTED,
        "miscaptioned": Label.MISCAPTIONED,
        "out-of-context": Label.OUT_OF_CONTEXT,
    }

    class_definitions = {
        Label.SUPPORTED:
            "The image and caption pair is truthful. This means the caption accurately "
            "describes the content of the image, providing correct and factual information.",
        Label.MISCAPTIONED:
            "The image and caption pair is miscaptioned. This means the caption provides incorrect "
            "information about the image content, misleading the viewer about what is depicted.",
        Label.OUT_OF_CONTEXT:
            "The image is used out of context. This means that while the caption may be factually correct, "
            "the image does not relate to the caption or is used in a misleading way to convey a false narrative."
    }

    extra_prepare_rules = """* **Identify Modality Balance**: Ensure that both text and image modalities are
    considered equally. Avoid unimodal bias by giving equal importance to the text and the associated image.
    * **Context Verification**: Since the images are sourced from various articles and Google Images, verify
    the context in which the image is used. This includes checking the source article and understanding the
    original context of the image.
    * **Prepare to Handle Real-World Data**: The data consists of real-world examples, so be prepared for
    diverse and potentially complex scenarios that may not be straightforward to classify."""

    extra_plan_rules = """* **Consider Both Modalities Equally**: Ensure that the verification process equally
    considers both the text and image modalities. Avoid focusing too much on one modality at the expense of the other.
    * **Plan for Multimodal Verification**: Develop a plan that includes steps for verifying both the text and
    the image. This might include object recognition, context checking, and text analysis.
    * **Check for Context and Misleading Information**: Verify the context of the image and caption. Check for
    any misleading information that might suggest the image is used out of context or the caption is miscaptioned.
    * **Identify Key Elements**: Break down the claim into its key components and identify which parts require
    verification. This includes identifying any potential asymmetry in the modalities.
    * **Reuse Previously Retrieved Knowledge**: Be prepared to reuse information and evidence gathered during
    previous verification steps to form a comprehensive judgment."""

    available_actions = None

    def __init__(self, variant="dev"):
        super().__init__(f"VERITE ({variant})", variant)
        self.file_path = Path(data_base_dir + "VERITE/VERITE.csv")
        self.data = self.load_data()

    def load_data(self) -> list[dict]:
        df = pd.read_csv(self.file_path)

        data = []
        for i, row in df.iterrows():
            image_path = Path(data_base_dir + f"VERITE/{row['image_path']}")
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                image = None
            entry = {
                "id": i,
                "content": Content(text=row["caption"], images=[image]),
                "label": self.class_mapping[row["label"]],
                "justification": row.get("ground_truth_justification", "")
            }
            data.append(entry)

        return data

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


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
