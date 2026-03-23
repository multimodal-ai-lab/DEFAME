from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL.Image import Image as PILImage
from ezmm import Image, MultimodalSequence

from config.globals import manipulation_detection_model
from defame.common import Results, Action
from defame.evidence_retrieval.tools.tool import Tool
from third_party.TruFor.src.fake_detect_tool import analyze_image, create_visualizations
from defame.prompts.prompts import SummarizeManipulationResultPrompt


class DetectManipulation(Action):
    """Detects manipulations of any kind within an image."""
    name = "detect_manipulation"
    requires_image = True

    def __init__(self, image: str):
        """
        @param image: The reference of the image to analyze.
        """
        self._save_parameters(locals())
        self.image = Image(reference=image)

    def __str__(self):
        return f'{self.name}({self.image.reference})'

    def __eq__(self, other):
        return isinstance(other, DetectManipulation) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


@dataclass
class ManipulationResults(Results):
    text: str = field(init=False)
    score: Optional[float]
    confidence_map: Image
    localization_map: Image
    noiseprint_map: Image

    def is_useful(self) -> Optional[bool]:
        return self.score is not None or self.confidence_map is not None

    def __str__(self):
        score_str = f'Score: {self.score:.3f}. (Everything above 0.6 might suggest manipulation.)' if self.score is not None else 'Score: N/A'
        loc_str = f'Localization map available: {self.localization_map.reference}.' if self.localization_map is not None else 'Localization map: N/A'
        conf_str = f'Confidence map available: {self.confidence_map.reference}.' if self.confidence_map is not None else 'Confidence map: N/A'
        noiseprint_str = f'Noiseprint++ available: {self.noiseprint_map.reference}.' if self.noiseprint_map is not None else 'Noiseprint++: N/A'
        #TODO: Perhaps include noiseprint for more precise manipulation detection.
        return f'Manipulation Detection Results\n{score_str}\n{loc_str}\n{conf_str}\n{noiseprint_str}'
    

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

    def _perform(self, action: DetectManipulation) -> ManipulationResults:
        result_dict = self.analyze_image(action.image.image)

        result = ManipulationResults(
            score=result_dict.get('score'),
            confidence_map=result_dict.get('confidence_map'),
            localization_map=result_dict.get('localization_map'),
            noiseprint_map=result_dict.get('noiseprint_map'),
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

    def _summarize(self, result: ManipulationResults, **kwargs) -> Optional[MultimodalSequence]:
        prompt = SummarizeManipulationResultPrompt(result) 
        summary = self.llm.generate(prompt)
        references = f"""
        Localization map at: {result.localization_map.reference}\n
        Confidence map at: {result.confidence_map.reference}\n
        Noiseprint map at: {result.noiseprint_map.reference}
        """
        return MultimodalSequence(f'{summary}\n{references}.')
