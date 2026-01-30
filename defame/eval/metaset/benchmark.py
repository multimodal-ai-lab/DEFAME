import json
import os
from pathlib import Path

from config.globals import data_root_dir
from defame.common import Label, Claim
from ezmm import Image
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Search, VerifyImageAuthenticity


class MetaSet(Benchmark):
    name = "MetaSet"
    shorthand = "metaset"

    is_multimodal = True

    # Map your ground_truth field to labels
    class_mapping = {
        "supported": Label.SUPPORTED,
        "refuted": Label.REFUTED,
    }

    class_definitions = {
        Label.SUPPORTED: (
            "An image-text pair is considered SUPPORTED when the "
            "image has not been manipulated (or is consistent with its metadata) "
            "and the text accurately describes the image."
        ),
        Label.REFUTED: (
            "An image-text pair is considered REFUTED when the claim is contradicted "
            "by the evidence, for example due to inconsistencies between the text and "
            "the image content or its metadata (date, time, device, author, location, etc.)."
        ),
    }
    extra_judge_rules = """
    * **Labeling**: If the evidence (especially metadata) contradicts the text, the verdict should be REFUTED.
      If text and image (including metadata) are consistent, the verdict should be SUPPORTED.
    """

    available_actions = [Search, VerifyImageAuthenticity]

    def __init__(self, variant="test"):
        self.base_image_path = data_root_dir
        super().__init__(variant, f"metaset/{variant}.json")

    def _load_data(self) -> list[dict]:
        with open(self.file_path, "r", encoding="utf-8-sig") as file:
            data = json.load(file)

        entries = []
        for ann in data:
            image_path = self.base_image_path / Path(ann["image"])
            if image_path and os.path.exists(image_path):
                image = Image(image_path)
                identifier = str(ann["id"])  # keep as string for consistency

                ground_truth = ann.get("ground_truth")  # "supported" / "refuted"
                claim_type = ann.get("type")            # e.g. "orig", "date", "time", "Meta_date", ...

                entry = {
                    "id": identifier,
                    "input": Claim(image, ann["text"], id=identifier),
                    "label": self.class_mapping.get(ground_truth, Label.REFUTED),
                    # >>> include the tags directly on the entry <<<
                    "ground_truth": ground_truth,
                    "type": claim_type,
                }
                entries.append(entry)
        return entries
