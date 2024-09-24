import os
from pathlib import Path
from typing import Iterator

import pandas as pd

from infact.common.medium import Image, media_registry
from config.globals import data_base_dir
from infact.common import Label, Content
from infact.eval.benchmark import Benchmark
from infact.tools.manipulation_detector import DetectManipulation
from infact.tools.text_extractor import OCR
from infact.tools.geolocator import Geolocate
from infact.tools.object_detector import DetectObjects
from infact.tools import WebSearch, ImageSearch, ReverseSearch


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
            "The image is used out of context. The caption is factual but the image is used in a misleading way to convey a false narrative."
    }

    extra_prepare_rules = """**Assess Alignment**: Assess the alignment between image and text in complex scenarios. Prepare for varied real-world images and captions.
    **Verify Context**: Examine the source and context of each image to understand its relationship with the accompanying text.
    Start each claim with: The Claim is that the image show ... """

    extra_plan_rules = """* **Consider Both Modalities Equally**: Avoid focusing too much on one modality at the expense of the other.
    * **Compare Image and Caption**: Verify the context of the image and caption. Check for
    any misleading information that might suggest the image is used out of context or the caption is miscaptioned.
    * **Identify any potential asymmetry in the modalities**: Perform one image_search if the action is available to compare other images with the claim image."""


    extra_judge_rules = """* **Focus on the alignment of Image and Claim**: The question is whether the image corresponds to the claim. 
    Judge if there is any alignment issue between image and text. Does the image deliver any support for the claim or is it taken out of context?
    If the claim is actually true but the image shows a different event, then the verdict is OUT OF CONTEXT. If the claim is false, then the verdict should be miscaptioned.
    Lastly, if the image appears to show the event mentioned in the claim, then the verdict is out-of-context."""

    available_actions = [WebSearch, DetectManipulation, DetectObjects, Geolocate, ImageSearch, ReverseSearch]

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
