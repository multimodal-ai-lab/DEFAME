import json
import os
from datetime import datetime
from pathlib import Path

from ezmm import Image

from config.globals import data_root_dir
from defame.common import Label, Claim
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Geolocate, Search


class ClaimReview2024(Benchmark):
    name = "ClaimReview2024+"
    shorthand = "claimreview"

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

    extra_plan_rules = """Always suggest the use of geolocation!"""

    available_actions = [Search, Geolocate]

    def __init__(self, variant="test"):
        super().__init__(variant, "ClaimReview2024/data_core.json")

    def _load_data(self) -> list[dict]:
        with open(self.file_path, "r") as f:
            raw_data = json.load(f)

        data = []
        for i, entry in enumerate(raw_data):
            image_path = Path(data_root_dir / "MAFC" / entry["claimImage"][0]) if entry["claimImage"] else None
            image = Image(image_path) if (image_path and os.path.exists(image_path)) else None
            claim = [image, entry["text"]] if image else entry["text"]
            label_text = entry.get("label")
            date = datetime.strptime(entry.get("claimDate"), "%Y-%m-%dT%H:%M:%SZ") if entry.get("claimDate") else None
            claim_entry = {
                "id": i,
                "input": Claim(claim,
                               id=i,
                               author=entry.get("author"),
                               date=date),
                "label": self.class_mapping.get(label_text),
            }
            data.append(claim_entry)

        return data


if __name__ == "__main__":
    benchmark = ClaimReview2024()
    for claim in benchmark:
        print(claim)
