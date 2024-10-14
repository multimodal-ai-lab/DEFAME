from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL.Image import Image as PILImage

from config.globals import manipulation_detection_model
from infact.common import MultimediaSnippet, Result, Action, Image
from infact.tools.tool import Tool
from third_party.TruFor.src.fake_detect_tool import analyze_image, create_visualizations


class DetectManipulation(Action):
    name = "detect_manipulation"
    description = "Detects manipulations within an image."
    how_to = "Provide an image and the model will analyze it for signs of manipulation."
    format = "detect_manipulation(<image:k>), where `k` is the image's ID"
    is_multimodal = True

    def __init__(self, image_ref: str):
        self.image: Image = MultimediaSnippet(image_ref).images[0]

    def __str__(self):
        return f'{self.name}({self.image.reference})'

    def __eq__(self, other):
        return isinstance(other, DetectManipulation) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


@dataclass
class ManipulationResult(Result):
    text: str = field(init=False)
    score: Optional[float]
    confidence_map: Optional[np.ndarray]
    localization_map: np.ndarray
    ref_confidence_map: Optional[str]
    ref_localization_map: Optional[str]
    noiseprint: Optional[np.ndarray] = None

    def is_useful(self) -> Optional[bool]:
        return self.score is not None or self.confidence_map is not None

    def __str__(self):
        score_str = f'Score: {self.score:.3f}. (Everything above 0.5 might suggest manipulation.)' if self.score is not None else 'Score: N/A'
        conf_str = f'Confidence map available: {self.ref_confidence_map}' if self.confidence_map is not None else 'Confidence map: N/A'
        loc_str = f'Localization map available: {self.ref_localization_map}' if self.localization_map is not None else 'Localization map: N/A'
        noiseprint_str = 'Noiseprint++ available' if self.noiseprint is not None else 'Noiseprint++: N/A'

        return f'Manipulation Detection Resultswi\n{score_str}\n{conf_str}\n{loc_str}\n{noiseprint_str}'

    def __post_init__(self):
        self.text = str(self)


class ManipulationDetector(Tool):
    name = "manipulation_detector"
    actions = [DetectManipulation]
    summarize = False

    def __init__(self, model_file: str = manipulation_detection_model, **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the Manipulation_Detector with a pretrained model.

        :param model_file: The path to the model file.
        :param device: The device to run the model on (e.g., "cpu" or "cuda").
        """
        self.model_file = model_file
        self.device = torch.device(self.device if self.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    def _perform(self, action: DetectManipulation) -> ManipulationResult:
        result_dict = self.analyze_image(action.image.image)

        result = ManipulationResult(
            score=result_dict.get('score'),
            confidence_map=result_dict.get('conf'),
            localization_map=result_dict['map'],
            noiseprint=result_dict.get('np++'),
            ref_localization_map=result_dict.get('ref_localization_map'),
            ref_confidence_map=result_dict.get('ref_confidence_map'),
        )
        return result

    def analyze_image(self, image: PILImage) -> dict:
        """
        :param image: A PIL image.
        :return: A dictionary containing the results of the manipulation detection.
        """
        result = analyze_image(image)
        result = create_visualizations(result)
        return result

    def _summarize(self, result: ManipulationResult, **kwargs) -> Optional[MultimediaSnippet]:
        return MultimediaSnippet("Manipulation Detector not fully implemented yet.")  # TODO: Implement the summary
