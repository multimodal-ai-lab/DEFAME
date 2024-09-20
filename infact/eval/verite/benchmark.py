import os
from pathlib import Path
from typing import Iterator

import pandas as pd

from infact.common.medium import Image, media_registry
from config.globals import data_base_dir
from infact.common import Label, Content
from infact.eval.benchmark import Benchmark
from infact.common.action import WebSearch, DetectManipulation, DetectObjects, Geolocate, OCR, ImageSearch


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
            "An image-caption pair is considered supported when the origin, content, and context of an image are accurately described in the accompanying caption.",
        Label.MISCAPTIONED:
            "An image-caption pair is considered miscaptioned when an image is being paired with a misleading caption that misrepresents the origin, content, and/or meaning of the image.",
        Label.OUT_OF_CONTEXT:
            "The image is used out of context when the caption does not relate to the image, or is used in a misleading way to convey a false narrative."
    }

    extra_prepare_rules = """**Assess Alignment**: Assess the alignment between image and text in complex scenarios. Prepare for varied real-world images and captions.
    **Verify Context**: Examine the source and context of each image to understand its relationship with the accompanying text."""

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

    available_actions = [WebSearch, DetectManipulation, DetectObjects, Geolocate, ImageSearch]

    def __init__(self, variant="dev"):
        super().__init__(f"VERITE ({variant})", variant)
        self.file_path = Path(data_base_dir + "VERITE/VERITE.csv")
        self.data = self.load_data()

    def load_data(self) -> list[dict]:
        df = pd.read_csv(self.file_path)

        data = []
        for i, row in df.iterrows():
            image_path = Path(data_base_dir + f"VERITE/{row['image_path']}")
            if not os.path.exists(image_path):
                continue  # TODO: Complete all missing images
            image = Image(image_path)
            image_ref = media_registry.add(image)
            assert isinstance(i, int)
            claim_text = f"{image_ref} {row['caption']}"
            entry = {
                "id": i,
                "content": Content(text=claim_text, id_number=i),
                "label": self.class_mapping[row["label"]],
                "justification": row.get("ground_truth_justification", "")
            }
            data.append(entry)

        return data

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)
