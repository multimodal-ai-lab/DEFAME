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
from PIL import Image
import pandas as pd

from common.claim import Claim
from common.label import Label
from common.shared_config import path_to_data
from safe.tools.search.wiki_dump import WikiDumpAPI
from common.modeling import Model


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
        self.is_multimodal = False
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

    def __init__(self, version=1, variant="dev"):
        super().__init__(f"fever{version}_{variant}")
        self.is_multimodal = False
        self.file_path = Path(path_to_data + f"FEVER/fever{version}_{variant}.jsonl")
        self.data = []
        self.load_data(version, variant, include_justifications)
        if version==1:
            self.extra_prepare_rules = """* **Identify the altered segments**: Since the Claim is generated by altering sentences from Wikipedia, pinpoint the parts of the Claim that seem modified or out of place.
* **Consider potential misleading elements**: The Claim may contain misleading or confusing elements due to the alteration process.
* **Prepare to investigate original context**: Since the Claim is derived from Wikipedia sentences, be prepared to trace back to the original context or related information for accurate verification."""

            self.extra_plan_rules = """* **Assume the Claim may be misleading**: Since the Claim is generated by altering Wikipedia sentences, consider that it might be intentionally misleading or designed to test the fact-checking process.
* **Identify key elements**: Break down the Claim into its key components and identify which parts require verification.
* **Plan for multiple investigation steps**: The Claim may require a series of verification steps, including checking the original Wikipedia context, cross-referencing related information, and verifying altered segments.
* **Consider alternative interpretations**: Given the altered nature of the Claim, consider multiple interpretations and verify each to ensure thorough fact-checking.
* **Reuse previously retrieved knowledge**: Be prepared to reuse information and evidence gathered during previous verification steps to form a comprehensive judgment."""

        if version==2:
            self.extra_prepare_rules = """* Before you start, begin with a _grammar check_ of the Claim. If it
has some grammatical errors, there is a high chance that the Claim means something different
than understandable at first glance. Take grammatical errors serious and elaborate on them.
* **Take the Claim literally**: Assume that each word of the Claim is as intended. Be strict
with the interpretation of the Claim.
* The Claim stems from a fact-checking challenge. A human fabricated the Claim artificially 
by using Wikipedia. The Claim could be a misleading prank, just like a trick question. It may also require
a chain of multiple investigation steps, re-using previously retrieved knowledge."""
            self.extra_plan_rules = """* The Claim stems from a fact-checking challenge. A human engineered the Claim
artificially by using Wikipedia. The Claim could be misleading, just like a trick question. It may
also require a chain of multiple investigation steps, re-using previously retrieved knowledge."""

    def load_data(self, version, variant, include_justifications=False):
        cache_file_path = os.path.join(path_to_data, f"FEVER/gt_justification_fever{version}_{variant}.jsonl")
        data = orjsonl.load(self.file_path)
        if include_justifications:
            self.data = self._load_with_justifications(data, cache_file_path, variant)
        else:
            self.data = self._load_without_justifications(data, variant)

    def _load_with_justifications(self, data, cache_file_path, variant):
        if os.path.exists(cache_file_path):
            ground_truth_data = self._load_ground_truth_data(cache_file_path, data)
        else:
            ground_truth_data = self._create_ground_truth_cache(data, cache_file_path)

        return self._populate_data_with_justifications(data, ground_truth_data, variant)

    def _load_without_justifications(self, data, variant):
        if variant in ["train", "dev"]:
            return [{"id": i,
                     "content": Claim(d["claim"]),
                     "label": self.class_mapping[d["label"].lower()],
                     "ground_truth_justification": None}
                    for i, d in enumerate(data)]
        else:
            return [{"id": i,
                     "content": Claim(d["claim"]),
                     "label": None,
                     "ground_truth_justification": None}
                    for i, d in enumerate(data)]

    def _load_ground_truth_data(self, cache_file_path, data):
        with open(cache_file_path, 'r') as cache_file:
            ground_truth_data = [json.loads(line) for line in cache_file]

        if len(ground_truth_data) < len(data):
            ground_truth_data.extend(self._update_ground_truth_cache(data, len(ground_truth_data), cache_file_path))

        return ground_truth_data

    def _create_ground_truth_cache(self, data, cache_file_path):
        ground_truth_data = []
        for i in tqdm(range(len(data)), desc="Creating cache"):
            ground_truth_data.append(self._get_entry_with_justification(i, data[i]))
        self._save_ground_truth_data(ground_truth_data, cache_file_path)
        return ground_truth_data

    def _update_ground_truth_cache(self, data, start_index, cache_file_path):
        new_entries = []
        for i in tqdm(range(start_index, len(data)), desc="Updating cache"):
            new_entries.append(self._get_entry_with_justification(i, data[i]))
        self._save_ground_truth_data(new_entries, cache_file_path, append=True)
        return new_entries

    def _save_ground_truth_data(self, ground_truth_data, cache_file_path, append=False):
        mode = 'a' if append else 'w'
        with open(cache_file_path, mode) as cache_file:
            for entry in ground_truth_data:
                cache_file.write(json.dumps(entry) + '\n')

    def _get_entry_with_justification(self, index, d):
        ground_truth_justification = self._extract_ground_truth_justification(d)
        return {
            "id": index,
            "content": d["claim"],
            "label": d["label"].lower(),
            "ground_truth_justification": ground_truth_justification
        }

    def _extract_ground_truth_justification(self, d):
        evidence_searcher = WikiDumpAPI()
        ground_truth_justification = []
        for evidence in d["evidence"][0] if d["evidence"] else [[None, None, None, None]]:
            evidence_key, relevant_sentence = evidence[2], evidence[3]
            if evidence_key and relevant_sentence is not None:
                evidence_text = evidence_searcher._call_api(evidence_key, 3)[0].text
                relevant_evidence = extract_nth_sentence(evidence_text, int(relevant_sentence))
                ground_truth_justification.append(relevant_evidence)
            else:
                ground_truth_justification.append(evidence_key)
        return ground_truth_justification

    def _populate_data_with_justifications(self, data, ground_truth_data, variant):
        if variant in ["train", "dev"]:
            return [{"id": i,
                     "content": Claim(d["claim"]),
                     "label": self.class_mapping[d["label"].lower()],
                     "ground_truth_justification": ground_truth_data[i]["ground_truth_justification"]}
                    for i, d in enumerate(data)]
        else:
            return [{"id": i,
                     "content": Claim(d["claim"]),
                     "label": None,
                     "ground_truth_justification": None}
                    for i, d in enumerate(data)]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)

class VERITE(Benchmark):
    class_mapping = {
        "true": Label.SUPPORTED,
        "miscaptioned": Label.REFUTED,
        "out-of-context": Label.OUT_OF_CONTEXT,
    }
    class_definitions = {
        Label.SUPPORTED:
            "The image and caption pair is truthful. This means the caption accurately "
            "describes the content of the image, providing correct and factual information.",
        Label.REFUTED:
            "The image and caption pair is miscaptioned. This means the caption provides incorrect " 
            "information about the image content, misleading the viewer about what is depicted.",
        Label.OUT_OF_CONTEXT:
            "The image is used out of context. This means that while the caption may be factually correct, "
            "the image does not relate to the caption or is used in a misleading way to convey a false narrative."
    }

    def __init__(self, variant="dev", justification_gen_LLM="gpt_4o"):
        super().__init__(f"verite_{variant}")
        self.is_multimodal = True
        self.file_path = Path(path_to_data + "VERITE/VERITE.csv")
        self.output_path = Path(path_to_data + f"VERITE/VERITE_w_justifications.csv")
        self.data = self.load_data()
        self.model = self.initialize_model(justification_gen_LLM)
        self.extra_prepare_rules = """* **Identify Modality Balance**: Ensure that both text and image modalities are considered equally. Avoid unimodal bias by giving equal importance to the text and the associated image.
* **Context Verification**: Since the images are sourced from various articles and Google Images, verify the context in which the image is used. This includes checking the source article and understanding the original context of the image.
* **Prepare to Handle Real-World Data**: The data consists of real-world examples, so be prepared for diverse and potentially complex scenarios that may not be straightforward to classify."""
        self.extra_plan_rules = """* **Consider Both Modalities Equally**: Ensure that the verification process equally considers both the text and image modalities. Avoid focusing too much on one modality at the expense of the other.
* **Plan for Multimodal Verification**: Develop a plan that includes steps for verifying both the text and the image. This might include object recognition, context checking, and text analysis.
* **Check for Context and Misleading Information**: Verify the context of the image and caption. Check for any misleading information that might suggest the image is used out of context or the caption is miscaptioned.
* **Identify Key Elements**: Break down the claim into its key components and identify which parts require verification. This includes identifying any potential asymmetry in the modalities.
* **Reuse Previously Retrieved Knowledge**: Be prepared to reuse information and evidence gathered during previous verification steps to form a comprehensive judgment."""


        if not os.path.exists(self.output_path):
            print("Generating justifications using model:", justification_gen_LLM)
            data = self.generate_justifications(self.data, self.model)
            self.data = self.save_data(data, self.output_path)
        else:
            print(f"Justifications file {self.output_path} already exists. Loading data.")
            self.data = self.load_data_with_justifications()

    def load_data(self):
        df = pd.read_csv(self.file_path)
        data = []

        for i, row in df.iterrows():
            image_path = Path(path_to_data + f"VERITE/{row['image_path']}")
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                image = None
            entry = {
                "id": i,
                "content": Claim(text=row["caption"], images=[image]),
                "label": self.class_mapping[row["label"]],
                "ground_truth_justification": row.get("ground_truth_justification", "")
            }
            data.append(entry)

        return data
    
    def load_data_with_justifications(self):
        df = pd.read_csv(self.output_path)
        data = []

        for i, row in df.iterrows():
            image_path = Path(path_to_data + f"VERITE/{row['image_path']}")
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                image = None
            entry = {
                "id": i,
                "content": Claim(text=row["caption"], images=[image]),
                "label": self.class_mapping[row["label"]],
                "ground_truth_justification": row["ground_truth_justification"]
            }
            data.append(entry)

        return data

    def initialize_model(self, model_name):
        model = Model(model_name)
        return model

    def generate_justifications(self, df, model):
        justifications = []

        for i in tqdm(range(0, len(df), 3), desc="Generating Justification Cache"):
            group = df.iloc[i:i+3]
            assert (group.iloc[0]["label"] == "true" and group.iloc[1]["label"] == "miscaptioned" and group.iloc[2]["label"] == "out-of-context")
            true_row = group[group["label"] == "true"]
            miscaptioned_row = group[group["label"] == "miscaptioned"]
            true_caption = true_row["caption"].values[0]
            false_caption = miscaptioned_row["caption"].values[0]

            # Generate justifications
            query = f"This is an image's true caption:'{true_caption}'. This is an image's manipulated caption: '{false_caption}'. Explain briefly how the miscaptioned image constitutes misinformation:"
            justification_false_caption = model.generate(query)
            justification_out_of_context = "The image is used out of context."

            df.loc[(df["caption"] == false_caption) & (df["label"] == "miscaptioned"), "ground_truth_justification"] = justification_false_caption
            df.loc[(df["caption"] == true_caption) & (df["label"] == "out-of-context"), "ground_truth_justification"] = justification_out_of_context

        return df

    def save_data(self, data, output_path):
        rows = []
        for entry in data:
            row = {
                "id": entry["id"],
                "caption": entry["content"].text,
                "image_path": entry["content"].images[0].filename,
                "label": entry["label"].name,
                "ground_truth_justification": entry["ground_truth_justification"]
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data.to_dict('records'))


def load_benchmark(name: str, **kwargs) -> Benchmark:
    match name:
        case "fever1":
            return FEVER(version=1, **kwargs)
        case "fever2":
            return FEVER(version=2, **kwargs)
        case "averitec":
            return AVeriTeC(**kwargs)
        case "verite":
            return VERITE(**kwargs)
        
def extract_nth_sentence(text, n):
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|\;)\s', text)
    
    # Ensure the index n is within the range of the sentences list
    if 0 <= n < len(sentences):
        return sentences[n]
    else:
        return "" 