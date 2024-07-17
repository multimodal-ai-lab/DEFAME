import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

from src.common.action import WebSearch
from src.common.content import Content
from src.common.label import Label
from config.globals import path_to_data
from src.eval.benchmark import Benchmark


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

    extra_judge_rules = """* **Do not commit the "argument from ignorance" fallacy**: The absence of evidence
    for claim `X` does NOT prove that `X` is false. Instead, `X` is simply unsupported--which is different
    to `X` being refuted. Unsupported yet not refuted claims are of the category `not enough information`."""

    available_actions = [WebSearch]

    def __init__(self, variant="dev"):
        super().__init__(f"AVeriTeC ({variant})")
        self.file_path = Path(path_to_data + f"AVeriTeC/{variant}.json")

        # Download the file if not done yet
        if not os.path.exists(self.file_path):
            url = f"https://huggingface.co/chenxwh/AVeriTeC/raw/main/data/{variant}.json"
            urlretrieve(url, self.file_path)

        # Load the data
        with open(self.file_path, 'r') as f:
            data_raw = json.load(f)

        data = []
        for i, d in enumerate(data_raw):
            content = Content(
                text=d["claim"],
                author=d["speaker"],
                date=datetime.strptime(d["claim_date"], "%d-%m-%Y"),
                origin=d["original_claim_url"]
            )
            label = self.class_mapping[d["label"]] if variant in ["train", "dev"] else None
            justification = d["justification"] if variant in ["train", "dev"] else None

            data.append({"id": i, "content": content, "label": label, "justification": justification})

        self.data = data

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)
