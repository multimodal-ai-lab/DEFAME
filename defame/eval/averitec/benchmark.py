import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from config.globals import data_root_dir, random_seed
from defame.common import Label, Content
from defame.tools import WebSearch, ImageSearch
from defame.eval.benchmark import Benchmark


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
            """The Claim has both supporting and refuting evidence from multiple sources or the Claim is 
            technically true but misleads by excluding important context.""",
    }

    extra_judge_rules = """Note that **the absence of evidence itself can be also regarded as
    evidence** if the context suggests that evidence should be present. This is particularly the
    case when the Claim implies news coverage or other observable consequences and sufficient time
    has passed for the evidence to emerge. In such a case, the Claim is likely to be refuted."""

    available_actions = [WebSearch]

    def __init__(self, variant: str ="dev"):
        super().__init__(name=f"AVeriTeC ({variant})", variant=variant)
        self.file_path = data_root_dir / f"AVeriTeC/{variant}.json"
        if not self.file_path.exists():
            raise ValueError(f"Unable to locate AVeriTeC at {data_root_dir.as_posix()}. "
                             f"See README.md for setup instructions.")

        # Load the data
        with open(self.file_path, 'r') as f:
            data_raw = json.load(f)

        data = []
        for i, d in enumerate(data_raw):
            date = d["claim_date"]
            content = Content(
                content=d["claim"],
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
