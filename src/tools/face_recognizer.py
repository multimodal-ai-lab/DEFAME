import torch

from src.common.action import FaceRecognition
from src.common.results import Result
from src.tools.tool import Tool


class FaceRecognizer(Tool):
    name = "face_recognizer"
    actions = [FaceRecognition]

    def __init__(self, model_name: str = "face-recognition-model", **kwargs):
        super().__init__(**kwargs)
        # self.model = pipeline("image-classification", model=model_name, device=device)
        self.model = None

    def perform(self, action: FaceRecognition) -> list[Result]:
        return [self.recognize_faces(action.image)]

    def recognize_faces(self, image: torch.Tensor) -> Result:
        # TODO: Implement this method
        # results = self.model(image)
        # faces = [result['label'] for result in results]
        # return faces
        text = "Face Recognition is not implemented yet."
        self.logger.log(str(text))
        result = Result()
        return result
