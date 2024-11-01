import os
from pathlib import Path
from typing import Iterator

import pandas as pd

from infact.common.medium import Image
from config.globals import data_base_dir, random_seed
from infact.common import Label, Content
from infact.eval.benchmark import Benchmark
from infact.tools.text_extractor import OCR
from infact.tools.geolocator import Geolocate
from infact.tools.object_detector import DetectObjects
from infact.tools import WebSearch, ImageSearch, ReverseSearch


class VERITE(Benchmark):
    shorthand = "verite"

    is_multimodal = True

    class_mapping = {
        "true": Label.SUPPORTED,
        "miscaptioned": Label.OUT_OF_CONTEXT,
        "out-of-context": Label.OUT_OF_CONTEXT,
    }

    class_definitions = {
        Label.SUPPORTED:
            "The claim accurately and coherently describes the origin, content, and context of the image.",
        Label.OUT_OF_CONTEXT:
            "The claim uses the image out of context, i.e., the image itself is pristine but the claim constructs a false narrative around the image or uses the image in a misleading way. In particular, the claim misrepresents the origin, content, and/or meaning of the image."
    }

    extra_prepare_rules = """**Assess Alignment**: Assess the alignment between image and text in complex scenarios. Prepare for varied real-world images and captions.
    **Verify Context**: Examine the source and context of each image to understand its relationship with the accompanying text.
    Start each claim with: The claim is that the image shows ... """

    extra_plan_rules = """* **Consider Both Modalities Equally**: Avoid focusing too much on one modality at the expense of the other but always check whether the text claim is true or false.
    * **Compare Image and Caption**: Verify the context of the image and caption.
    * **Identify any potential asymmetry in the modalities**: Perform one image_search if the action is available to compare other images with the claim image."""

    extra_judge_rules = """* **Caption Check First**: If the caption is factually wrong, then the claim is considered miscaptioned.
    * **Alignment Check of Image and Claim**: If the caption is factually correct, we need to check whether the image corresponds to the claim. 
    Judge if there is any alignment issue between image and text. Does the image deliver any support for the claim or is it taken out of context?
    If the claim text is actually true but the image shows a different event, then the verdict is out-of-context."""

    available_actions = [WebSearch, Geolocate, ImageSearch, ReverseSearch]

    def __init__(self, variant="dev", n_samples: int=None):
        super().__init__(f"VERITE ({variant})", variant)
        self.file_path = Path(data_base_dir + "VERITE/VERITE.csv")
        self.data = self.load_data(n_samples)

    def load_data(self, n_samples: int=None) -> list[dict]:
        # TODO: Increase efficiency
        df = pd.read_csv(self.file_path)
        # TODO: Adhere to random_shuffle hyperparam and do not loose instances while building
        # if n_samples and (n_samples < len(df)):
        #     df = df.sample(n=n_samples, random_state=random_seed)
        data = []
        for i, row in df.iterrows():
            image_path = Path(data_base_dir + f"VERITE/{row['image_path']}")
            if not os.path.exists(image_path):
                continue  # TODO: Complete all missing images
            image = Image(image_path)
            assert isinstance(i, int)
            claim_text = f"{image.reference} {row['caption']}"
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
