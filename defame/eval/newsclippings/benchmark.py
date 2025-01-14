import json
import os
from pathlib import Path
from typing import Iterator

from defame.common.medium import Image
from config.globals import data_root_dir, random_seed
from defame.common import Label, Content
from defame.eval.benchmark import Benchmark
from defame.tools.text_extractor import OCR
from defame.tools.geolocator import Geolocate
from defame.tools.object_detector import DetectObjects
from defame.tools import WebSearch, ImageSearch, ReverseSearch


class NewsCLIPpings(Benchmark):
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

    available_actions = [WebSearch, Geolocate, ImageSearch, ReverseSearch]

    def __init__(self, variant="val", n_samples: int=None):
        super().__init__(f"NewsCLIPpings ({variant})", variant)
        self.visual_news_file_path = data_root_dir / "NewsCLIPings/news_clippings/visual_news/origin/data.json"
        self.data_file_path = data_root_dir / f"NewsCLIPings/news_clippings/news_clippings/data/merged_balanced/{variant}.json"
        if not self.data_file_path.exists():
            raise ValueError(f"Unable to locate NewsCLIPpings at {data_root_dir.as_posix()}. "
                             f"See README.md for setup instructions.")

        self.visual_news_data_mapping = self.load_visual_news_data()
        self.data = self.load_data(n_samples)

    def load_visual_news_data(self) -> dict:
        """Load visual news data and map it by ID."""
        with open(self.visual_news_file_path, "r", encoding="utf-8") as file:
            visual_news_data = json.load(file)
        return {ann["id"]: ann for ann in visual_news_data}

    def load_data(self, n_samples: int=None) -> list[dict]:
        """Load annotations data from the NewsCLIPings dataset and map captions to images."""
        with open(self.data_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        annotations = data["annotations"]

        # Pre-select instances
        # TODO: Add random sampling
        if n_samples:
            annotations = annotations[:n_samples]

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
                claim_text = f"{image.reference} {caption}"
                # id = f'{ann_data["id"]}_{ann_data["image_id"]}'
                entry = {
                    "id": i,
                    "content": Content(content=claim_text,
                                       id_number=i,
                                       meta_info="Published: some date between 2005 and 2020."),
                    "label": self.class_mapping[str(ann["falsified"])],
                    "justification": ann.get("justification", "")
                }
                entries.append(entry)

        return entries

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)
