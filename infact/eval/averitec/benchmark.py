import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from config.globals import data_base_dir
from infact.common import Label, Content
from infact.tools import WebSearch, ImageSearch
from infact.eval.benchmark import Benchmark


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

    available_actions = [WebSearch, ImageSearch]

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
