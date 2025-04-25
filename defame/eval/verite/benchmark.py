import os

import pandas as pd
from ezmm import Image

from config.globals import data_root_dir
from defame.common import Label, Claim
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Geolocate, Search


class VERITE(Benchmark):
    name = "VERITE"
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

    extra_judge_rules = """* **Caption Check First**: If the caption is factually wrong, then the claim is considered out-of-context.
    * **Alignment Check of Image and Claim**: If the caption is factually correct, we need to check whether the image corresponds to the claim. 
    Judge if there is any alignment issue between image and text. Does the image deliver any support for the claim or is it taken out of context?
    If the claim text is actually true but the image shows a different event, then the verdict is out-of-context."""

    available_actions = [Search, Geolocate]

    def __init__(self, variant="dev"):
        super().__init__(variant, "VERITE/VERITE.csv")

    def _load_data(self) -> list[dict]:
        # TODO: Increase efficiency
        df = pd.read_csv(self.file_path)
        data = []
        for i, row in df.iterrows():
            image_path = data_root_dir / f"VERITE/{row['image_path']}"
            if not os.path.exists(image_path):
                continue
            image = Image(image_path)
            identifier = str(i)
            entry = {
                "id": identifier,
                "input": Claim(image, row['caption'], id=identifier),
                "label": self.class_mapping[row["label"]],
                "justification": row.get("ground_truth_justification", "")
            }
            data.append(entry)

        return data
