import torch
from PIL import Image

from config.globals import manipulation_detection_model
from infact.common.results import ManipulationResult, Evidence
from infact.tools.tool import Tool
from infact.common.action import DetectManipulation
from third_party.TruFor.test_docker.src.fake_detect_tool import analyze_image

class Manipulation_Detector(Tool):
    name = "manipulation_detector"
    actions = [DetectManipulation]

    def __init__(self, model_file: str = manipulation_detection_model, **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the Manipulation_Detector with a pretrained model.

        :param model_file: The path to the model file.
        :param device: The device to run the model on (e.g., "cpu" or "cuda").
        """
        self.model_file = model_file
        self.device = torch.device(self.device if self.device else ('cuda' if torch.cuda.is_available() else 'cpu'))


    def perform(self, action: DetectManipulation) -> Evidence:
        result_dict = self.analyze_image(action.image)

        result = ManipulationResult(
                score=result_dict.get('score'),
                confidence_map=result_dict.get('conf'),
                localization_map=result_dict['map'],
                noiseprint=result_dict.get('np++')
            )
        return Evidence(str(result), [result])

    def analyze_image(self, image: Image.Image) -> dict:
            """
            :param image: A PIL image.
            :return: A dictionary containing the results of the manipulation detection.
            """
            result = analyze_image(image)
            return result
    


