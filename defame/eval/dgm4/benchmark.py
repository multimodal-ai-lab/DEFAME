import json
import os
from pathlib import Path
from typing import Iterator

from defame.common.medium import Image
from config.globals import data_root_dir
from defame.common import Label, Content
from defame.eval.benchmark import Benchmark
from defame.tools import WebSearch, ImageSearch, ReverseSearch


class DGM4(Benchmark):
    shorthand = "dgm4"
    
    is_multimodal = True

    class_mapping = {
        "orig": Label.SUPPORTED,
        "face_swap": Label.REFUTED,
        "object_swap": Label.REFUTED,
        "background_swap": Label.REFUTED,
        "scene_swap": Label.REFUTED,
        # Add other classes as needed
    }

    class_definitions = {
        Label.SUPPORTED: "An image-text pair is considered original (SUPPORTS the claim) when the image has not been manipulated and the text accurately describes the image",
        Label.REFUTED: "An image-text pair is considered falsified when the image has been manipulated (e.g., face swap, object swap, etc.) or the text does not align with the image."
    }

    extra_prepare_rules = """
    **Assess Manipulation**: Assess the presence of harmful manipulations in both the image and text. Pay attention to the global manipulations (Swap) and local manipulations (Attribute) in both modalities.
    **Verify Manipulation Type**: Identify whether the manipulation is global (e.g., face or text swap) or fine-grained (e.g., face attribute or text sentiment). For each type, examine how the manipulation impacts the overall claim.
    Start each claim with: The claim is that the image and text correspond to ... 
    """
    extra_plan_rules = """
    * **Compare Image and Text**: Evaluate how the image and the text interact. Check whether the image's manipulation affects the interpretation of the text and vice versa.
    * **Investigate Face Manipulations**: For manipulations like face swaps or attribute changes, assess whether they align with the corresponding text. Ensure face swaps and altered facial attributes are correctly identified.
    * **Examine Text Manipulations**: Assess semantic or sentiment-based manipulations in the text. Pay special attention to cases where the named entity remains the same but the surrounding context changes.
    * **Use Cross-Modality Tools**: Utilize available tools e.g., manipulation_detection(), reverse_search() and image search() to cross-check whether the manipulations in one modality (image or text) create inconsistencies with the other.
    * **Always use detect_manipulation() to identify potential manipulation of the image.
    """
    extra_judge_rules = """
    * **Classify based on Manipulation Type**: If the image or text shows manipulations (either global or fine-grained), the verdict should be REFUTED.
    * **Text Manipulations**: For text-based manipulations, focus on semantic changes or sentiment shifts. If these changes distort the original meaning, the verdict should be REFUTED.
    """

    available_actions = [WebSearch, ImageSearch, ReverseSearch]

    def __init__(self, variant="train"):
        super().__init__(f"DGM4 ({variant})", variant)
        self.data_file_path = data_root_dir / f"DGM4/metadata/{variant}.json"
        if not self.file_path.exists():
            raise ValueError(f"Unable to locate DGM4 at {data_root_dir.as_posix()}. "
                             f"See README.md for setup instructions.")

        self.base_image_path = data_root_dir
        self.data = self.load_data() #TODO: Shift the sampling to the parent class Benchmark like a (sample() function)

    def load_data(self) -> list[dict]:
        """Load the DGM4 annotations and construct the entries."""
        with open(self.data_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        entries = []
        for i, ann in enumerate(data):
            # Construct the image path
            image_path = self.base_image_path / Path(ann["image"])
            
            # Ensure the image path exists
            if image_path and os.path.exists(image_path):
                image = Image(image_path)
                claim_text = f'{ann["text"]} {image.reference}'
                id = ann["id"]
                entry = {
                    "id": id,
                    "content": Content(content=claim_text, id_number=id),
                    "label": self.class_mapping.get(ann["fake_cls"], Label.REFUTED),  # Map fake_cls to label
                    "justification": f'The image manipulation class is {ann["fake_cls"]}.'
                }
                entries.append(entry)

        return entries

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)
