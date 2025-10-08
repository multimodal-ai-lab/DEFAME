import json
import os
from pathlib import Path

from config.globals import data_root_dir
from defame.common import Label, Claim
from ezmm import Image
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Search, DetectDeepfake
from defame.evidence_retrieval.tools.manipulation_detector import DetectManipulation


class DGM4(Benchmark):
    name = "DGM4"
    shorthand = "dgm4"

    is_multimodal = True

    class_mapping = {
        "orig": Label.SUPPORTED,
        "face_swap": Label.REFUTED,
        "object_swap": Label.REFUTED,
        "background_swap": Label.REFUTED,
        "scene_swap": Label.REFUTED,
        "text_swap": Label.REFUTED,
        "face_attribute": Label.REFUTED,
        "text_attribute": Label.REFUTED,
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
    * **Use Cross-Modality Tools**: Utilize available tools e.g., deepfake_detection(), reverse_search() and image search() to cross-check whether the manipulations in one modality (image or text) create inconsistencies with the other.
    """
    extra_judge_rules = """
    * **Classify based on Manipulation Type**: If the image or text shows manipulations (either global or fine-grained), the verdict should be REFUTED. If the deepfake detector show a really low probability of fakeness like 0.51 then consider the image real and depend on the the other tools to detect the verdict . 
    * **Text Manipulations**: For text-based manipulations, focus on semantic changes or sentiment shifts. If these changes distort the original meaning, the verdict should be REFUTED.
    """

    available_actions = [Search, DetectDeepfake]

    def __init__(self, variant="val"):
        self.base_image_path = data_root_dir
        # fast lookup used later to enrich predictions.csv
        self.id2fake_cls = {}
        super().__init__(variant, f"DGM4/metadata/{variant}.json")

    def _load_data(self) -> list[dict]:
        with open(self.file_path, "r", encoding="utf-8-sig") as file:
            data = json.load(file)

        entries = []
        for ann in data:
            image_path = self.base_image_path / Path(ann["image"])
            if image_path and os.path.exists(image_path):
                image = Image(image_path)
                identifier = str(ann["id"])  # keep as string for consistency
                fake_cls = ann.get("fake_cls")
                # cache for later, so finalize_evaluation can add it to predictions.csv
                self.id2fake_cls[identifier] = fake_cls

                entry = {
                    "id": identifier,
                    "input": Claim(image, ann["text"], id=identifier),
                    "label": self.class_mapping.get(fake_cls, Label.REFUTED),
                    # >>> include the category directly on the entry <<<
                    "fake_cls": fake_cls,
                }
                entries.append(entry)
        return entries
