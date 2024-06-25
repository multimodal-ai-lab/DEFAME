import json
import random
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import MutableSequence, Iterable, Iterator
import re
import os
import orjsonl
from tqdm import tqdm

from common.claim import Claim
from common.label import Label
from common.shared_config import path_to_data
from safe.tools.search.wiki_dump import WikiDumpAPI


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
        if variant in ["train", "dev"]:
            self.data = [{"id": i,
                          "content": Claim(
                              text=d["claim"],
                              author=d["speaker"],
                              date=datetime.strptime(d["claim_date"], "%d-%m-%Y"),
                              origin=d["original_claim_url"]
                          ),
                          "label": self.class_mapping[d["label"]],
                          "ground_truth_justification": d["justification"]}
                         for i, d in enumerate(data)]
        else:
            self.data = [{"id": i,
                          "content": Claim(
                              text=d["claim"],
                              author=d["speaker"],
                              date=datetime.strptime(d["claim_date"], "%d-%m-%Y"),
                              origin=d["original_claim_url"]
                          ),
                          "label": None,
                          "ground_truth_justification": None}
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

    def __init__(self, version=1, variant="dev"):
        super().__init__(f"fever{version}_{variant}")
        self.file_path = Path(path_to_data + f"FEVER/fever{version}_{variant}.jsonl")
        # Load the data 
        self.data = []
        self.load_data(version, variant)
        

    def load_data(self, version, variant, include_justifications=False):
        
        
        cache_file_path = os.path.join(path_to_data, f"FEVER/gt_justification_fever{version}_{variant}.jsonl")
        data = orjsonl.load(self.file_path)
        evidence_searcher = WikiDumpAPI()

        if include_justifications:
            if os.path.exists(cache_file_path):
                # Load ground truth justifications from the cache file
                with open(cache_file_path, 'r') as cache_file:
                    ground_truth_data = [json.loads(line) for line in cache_file]

                # Check if the number of lines in the cache file matches the number of samples
                if len(ground_truth_data) < len(data):
                    print("Updating ground truth justifications.")
                    for i in tqdm(range(len(ground_truth_data), len(data)), desc="Updating cache"):
                        d = data[i]
                        ground_truth_justification = []

                        for evidence in d["evidence"][0] if d["evidence"] else [[None, None, None, None]]:
                            evidence_key, relevant_sentence = evidence[2], evidence[3]

                            if evidence_key and relevant_sentence is not None:
                                evidence_text = evidence_searcher._call_api(evidence_key, 3)[0].text
                                relevant_evidence = extract_nth_sentence(evidence_text, int(relevant_sentence))
                                ground_truth_justification.append(relevant_evidence)
                            else:
                                ground_truth_justification.append(evidence_key)

                        entry = {
                            "id": i,
                            "content": Claim(d["claim"]),
                            "label": self.class_mapping[d["label"].lower()],
                            "ground_truth_justification": ground_truth_justification
                        }
                        self.data.append(entry)
                        ground_truth_data.append(entry)

                        # Append the new entries to the cache file
                        with open(cache_file_path, 'a') as cache_file:
                            for entry in ground_truth_data[len(ground_truth_data) - (len(data) - len(ground_truth_data)):]:
                                cache_file.write(json.dumps(entry) + '\n')
            else:
                print("Creating ground truth justifications.")
                ground_truth_data = []

                for i in tqdm(range(len(data)), desc="Creating cache"):
                    d = data[i]
                    ground_truth_justification = []

                    for evidence in d["evidence"][0] if d["evidence"] else [[None, None, None, None]]:
                        evidence_key, relevant_sentence = evidence[2], evidence[3]

                        if evidence_key and relevant_sentence is not None:
                            evidence_text = evidence_searcher._call_api(evidence_key, 3)[0].text
                            relevant_evidence = extract_nth_sentence(evidence_text, int(relevant_sentence))
                            ground_truth_justification.append(relevant_evidence)
                        else:
                            ground_truth_justification.append(evidence_key)

                    entry = {
                        "id": i,
                        "content": Claim(d["claim"]),
                        "label": self.class_mapping[d["label"].lower()],
                        "ground_truth_justification": ground_truth_justification
                    }
                    self.data.append(entry)
                    ground_truth_data.append(entry)
            
            # Save the generated ground truth justifications to the cache file
            with open(cache_file_path, 'w') as cache_file:
                for entry in ground_truth_data:
                    cache_file.write(json.dumps(entry) + '\n')

            # Populate self.data from ground_truth_data
            if variant in ["train", "dev"]:
                for i, d in enumerate(data):
                    entry = {
                        "id": i,
                        "content": Claim(d["claim"]),
                        "label": self.class_mapping[d["label"].lower()],
                        "ground_truth_justification": ground_truth_data[i]["ground_truth_justification"]
                    }
                    self.data.append(entry)
            else:
                self.data = [{"id": i,
                              "content": Claim(d["claim"]),
                              "label": None,
                              "ground_truth_justification": None
                              }
                          for i, d in enumerate(data)]
        else:
            if variant in ["train", "dev"]:
                for i, d in enumerate(data):
                    entry = {
                        "id": i,
                        "content": Claim(d["claim"]),
                        "label": self.class_mapping[d["label"].lower()],
                        "ground_truth_justification": None
                    }
                    self.data.append(entry)
            else:
                self.data = [{"id": i,
                              "content": Claim(d["claim"]),
                              "label": None,
                              "ground_truth_justification": None
                              }
                          for i, d in enumerate(data)]


class FEVEROUS(Benchmark):
    #TODO this is not ready yet
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
        super().__init__(f"feverous_{variant}")
        self.file_path = Path(path_to_data + f"Feverous/feverous_{variant}_challenges.jsonl")
        # Load the data 
        self.data = []
        self.load_data(variant)
        

    def load_data(self, variant):
        cache_file_path = os.path.join(path_to_data, f"Feverous/gt_justification_{variant}.jsonl")

        data = orjsonl.load(self.file_path)
        evidence_searcher = WikiDumpAPI()

        if os.path.exists(cache_file_path):
            # Load ground truth justifications from the cache file
            with open(cache_file_path, 'r') as cache_file:
                ground_truth_data = [json.loads(line) for line in cache_file]

            # Check if the number of lines in the cache file matches the number of samples
            if len(ground_truth_data) < len(data):
                print("Updating ground truth justifications.")
                for i in tqdm(range(len(ground_truth_data), len(data)), desc="Updating cache"):
                    d = data[i]
                    ground_truth_justification = []

                    for evidence in d["evidence"][0] if d["evidence"] else [[None, None, None, None]]:
                        evidence_key, relevant_sentence = evidence[2], evidence[3]

                        if evidence_key and relevant_sentence is not None:
                            evidence_text = evidence_searcher._call_api(evidence_key, 3)[0].text
                            relevant_evidence = extract_nth_sentence(evidence_text, int(relevant_sentence))
                            ground_truth_justification.append(relevant_evidence)
                        else:
                            ground_truth_justification.append(evidence_key)

                    entry = {
                        "id": i,
                        "content": Claim(d["claim"]),
                        "label": self.class_mapping[d["label"].lower()],
                        "ground_truth_justification": ground_truth_justification
                    }
                    self.data.append(entry)
                    ground_truth_data.append(entry)
                
                # Append the new entries to the cache file
                with open(cache_file_path, 'a') as cache_file:
                    for entry in ground_truth_data[len(ground_truth_data) - (len(data) - len(ground_truth_data)):]:
                        cache_file.write(json.dumps(entry) + '\n')
        else:
            print("Creating ground truth justifications.")
            ground_truth_data = []

            for i in tqdm(range(1,len(data)), desc="Creating cache"):
                d = data[i]
                ground_truth_justification = []

                for evidence in d["evidence"]["content"] if d["evidence"] else [[None, None, None, None]]:
                    evidence_key, relevant_sentence = evidence[2], evidence[3]

                    if evidence_key and relevant_sentence is not None:
                        evidence_text = evidence_searcher._call_api(evidence_key, 3)[0].text
                        relevant_evidence = extract_nth_sentence(evidence_text, int(relevant_sentence))
                        ground_truth_justification.append(relevant_evidence)
                    else:
                        ground_truth_justification.append(evidence_key)

                entry = {
                    "id": i,
                    "content": Claim(d["claim"]),
                    "label": self.class_mapping[d["label"].lower()],
                    "ground_truth_justification": ground_truth_justification
                }
                self.data.append(entry)
                ground_truth_data.append(entry)
            
            # Save the generated ground truth justifications to the cache file
            with open(cache_file_path, 'w') as cache_file:
                for entry in ground_truth_data:
                    cache_file.write(json.dumps(entry) + '\n')

        # Populate self.data from ground_truth_data
        if variant in ["train", "dev"]:
            for i, d in enumerate(data):
                entry = {
                    "id": i,
                    "content": Claim(d["claim"]),
                    "label": self.class_mapping[d["label"].lower()],
                    "ground_truth_justification": ground_truth_data[i]["ground_truth_justification"]
                }
                self.data.append(entry)
        else:
            self.data = [{"id": i,
                          "content": Claim(d["claim"]),
                          "label": None,
                          "ground_truth_justification": None
                          }
                      for i, d in enumerate(data)]


def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


def load_benchmark(name: str, **kwargs) -> Benchmark:
    match name:
        case "fever1":
            return FEVER(version=1, **kwargs)
        case "fever2":
            return FEVER(version=2, **kwargs)
        case "averitec":
            return AVeriTeC(**kwargs)
        case "feverous":
            return FEVEROUS(**kwargs)
        
def extract_nth_sentence(text, n):
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|\;)\s', text)
    
    # Ensure the index n is within the range of the sentences list
    if 0 <= n < len(sentences):
        return sentences[n]
    else:
        return "" 