import json
import os
from pathlib import Path

from ezmm import Image

from config.globals import data_root_dir
from defame.common import Label, Claim
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Geolocate, Search


class NewsCLIPpings(Benchmark):
    name = "NewsCLIPpings"
    shorthand = "newsclippings"

    is_multimodal = True

    class_mapping = {
        "False": Label.SUPPORTED,
        "True": Label.REFUTED,  # Mapping 'falsified' to REFUTED
    }

    class_definitions = {
        Label.SUPPORTED:
            "An image-caption pair is considered true when the origin, content, and context of an image are accurately described in the accompanying caption.",
        Label.REFUTED:
            """An image-caption pair is considered falsified when an image is paired with a misleading caption that misrepresents the origin, content, or meaning of the image.
             If the claim is paired with an image with similar semantics to the caption but the named entities are disjoint or they refer to different people or events, the verdict is also "falsified"."""
    }

    extra_prepare_rules = """**Assess Alignment**: Assess the alignment between image and text in complex scenarios, especially where images are swapped or altered to mislead. Prepare for diverse real-world images and captions that may intentionally misrepresent events.
    **Verify Context**: Examine the source, origin, and context of each image to understand how it relates to the accompanying caption. Identify where an image may have been substituted to convey a false narrative.
    Start each claim with: The claim is that the image shows ... """

    extra_plan_rules = """* Ensure that both the text and image are evaluated together, as discrepancies often arise from intentional misalignment of the two.
    * **Compare Image and Caption**: Check for any misleading information, especially in cases where the image does not semantically align with the caption. Pay attention to subtle changes in meaning.
    * **Identify Swapped Images**: Investigate whether the image has been swapped with one from a different event or with different individuals. Utilize tools like image_search to cross-check the image context."""

    extra_judge_rules = """* The main goal is to verify whether the image supports the claim in the caption or has been used to mislead by being swapped or taken out of context. 
    * If the image has been used from a different event or misrepresents individuals, the verdict should be REFUTED.
    * If the image and caption match the event, the verdict is SUPPORTED.
    * If you do not find any apparent contradictions then the sample is probably SUPPORTED."""

    base_path = data_root_dir / "NewsCLIPpings/news_clippings/visual_news/origin/"

    available_actions = [Search, Geolocate]

    def __init__(self, variant="val", n_samples: int = None):
        self.n_samples = n_samples
        super().__init__(variant, f"NewsCLIPings/news_clippings/news_clippings/data/merged_balanced/{variant}.json")
        self.visual_news_file_path = data_root_dir / "NewsCLIPings/news_clippings/visual_news/origin/data.json"
        self.visual_news_data_mapping = self.load_visual_news_data()

    def load_visual_news_data(self) -> dict:
        """Load visual news data and map it by ID."""
        with open(self.visual_news_file_path, "r", encoding="utf-8") as file:
            visual_news_data = json.load(file)
        return {ann["id"]: ann for ann in visual_news_data}

    def _load_data(self) -> list[dict]:
        """Load annotations data from the NewsCLIPings dataset and map captions to images."""
        # TODO: Load benchmark lazily

        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        annotations = data["annotations"]

        # Pre-select instances
        # TODO: Maybe perform random sampling before selection
        if self.n_samples:
            annotations = annotations[:self.n_samples]

        entries = []
        for i, ann in enumerate(annotations):
            # Map the caption and image paths using the visual_news_data_mapping
            caption = self.visual_news_data_mapping.get(ann["id"], {}).get("caption")
            if not caption:
                continue

            relative_path = self.visual_news_data_mapping.get(ann["image_id"], {}).get("image_path")
            image_path = self.base_path / Path(relative_path[2:])

            if image_path and os.path.exists(image_path):
                image = Image(image_path)
                entry = {
                    "id": str(i),
                    "input": Claim(image, caption,
                                   id=str(i),
                                   meta_info="Published: some date between 2005 and 2020."),
                    "label": self.class_mapping[str(ann["falsified"])],
                    "justification": ann.get("justification", "")
                }
                entries.append(entry)

        return entries
