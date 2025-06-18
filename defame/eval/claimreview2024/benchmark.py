import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator

from huggingface_hub import snapshot_download, login, HfFolder

from config.globals import data_root_dir, api_keys
from defame.common import Label, Content
from defame.common.medium import Image
from defame.eval.benchmark import Benchmark
from defame.tools import WebSearch, ImageSearch, ReverseSearch, Geolocate


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

    extra_plan_rules = """Always suggest the use of geolocation!"""

    available_actions = [WebSearch, Geolocate, ImageSearch, ReverseSearch]

    def __init__(self, variant="dev", json_file="data.json"):
        super().__init__(f"ClaimReview2024+ Benchmark ({variant})", variant)
        self.file_path = Path(data_root_dir / "ClaimReview2024plus/test.json")

        if not self.file_path.exists():
            # Log in (if not logged in yet)
            if not HfFolder.get_token():
                if hf_token := api_keys["huggingface_user_access_token"]:
                    login(hf_token)

            # Download the dataset from Hugging Face:
            # Ensure you are logged in via `huggingface-cli login` and have
            # got access to the dataset
            snapshot_download(repo_id="MAI-Lab/ClaimReview2024plus",
                              repo_type="dataset",
                              local_dir=self.file_path.parent)

        self.data = self.load_data()

    def load_data(self) -> list[dict]:
        with open(self.file_path, "r") as f:
            raw_data = json.load(f)

        data = []
        for i, entry in enumerate(raw_data):
            image_path = Path(data_root_dir / "MAFC" / entry["image"][0]) if entry["image"] else None
            image = Image(image_path) if (image_path and os.path.exists(image_path)) else None
            claim_text = f"{image.reference} {entry['text']}" if image else f"{entry['text']}"
            label_text = entry.get("label")
            date_str = entry.get("date")
            date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ") if date_str else None
            claim_entry = {
                "id": i,
                "content": Content(content=claim_text,
                                   id_number=i,
                                   author=entry.get("author"),
                                   date=date),
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
