import os
import json
from pathlib import Path
from typing import Iterator
from datetime import datetime

from defame.common.medium import Image
from config.globals import data_root_dir
from defame.common import Label, Content
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Geolocate, WebSearch, ImageSearch, ReverseSearch


class ClaimReview2024(Benchmark):
    shorthand = "claimreview2024"

    is_multimodal = True

    class_mapping = {
        "refuted": Label.REFUTED,
        "supported": Label.SUPPORTED,
        "not enough information": Label.NEI,
        "misleading": Label.CHERRY_PICKING,
    }

    class_definitions = {
        Label.SUPPORTED: "The claim is accurate based on evidence.",
        Label.REFUTED: "A claim is considered refuted when the evidence contradicts the claim.",
        Label.CHERRY_PICKING: "The claim is misleading or requires additional context.",
        Label.NEI: "The claim does not have enough information to be verified.",
    }
    
    #extra_prepare_rules = """**Assess Alignment**: Check the relationship between text and image in claims.
    #**Verify Context**: Ensure proper attribution and context for each image."""

    extra_plan_rules = """Always suggest the use of geolocation!"""

    #extra_judge_rules = """* **Context and Accuracy Check**: Determine if the text matches the image and sources.
    #* **Alignment Evaluation**: Ensure the image supports the textual claim."""

    available_actions = [WebSearch, Geolocate, ImageSearch, ReverseSearch]

    def __init__(self, variant="dev", json_file="data.json"):
        super().__init__(f"ClaimReview2024+ Benchmark ({variant})", variant)
        self.file_path = Path(data_root_dir / "ClaimReview2024/data_core.json")
        self.data = self.load_data()

    def load_data(self) -> list[dict]:
        with open(self.file_path, "r") as f:
            raw_data = json.load(f)
        
        data = []
        for i, entry in enumerate(raw_data):
            image_path = Path(data_root_dir / "MAFC" /entry["claimImage"][0]) if entry["claimImage"] else None
            image = Image(image_path) if (image_path and os.path.exists(image_path)) else None
            claim_text = f"{image.reference} {entry['text']}" if image else f"{entry['text']}"
            label_text = entry.get("label")
            origin = entry["claimReview"][0]["url"] if entry.get("claimReview") else None
            date = datetime.strptime(entry.get("claimDate"), "%Y-%m-%dT%H:%M:%SZ") if entry.get("claimDate") else None
            claim_entry = {
                "id": i,
                "content": Content(content=claim_text, 
                                   id_number=i, 
                                   #author=entry.get("claimant"), 
                                   date=date,
                                   ),
                "label": self.class_mapping.get(label_text),
                "justification": "",
            }
            data.append(claim_entry)

        return data

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


if __name__ == "__main__":
    benchmark = ClaimReview2024(variant="test", json_file="data.json")
    for claim in benchmark:
        print(claim)
