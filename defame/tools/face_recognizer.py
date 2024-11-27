from typing import Optional

import torch

from defame.common import MultimediaSnippet, Action, Image, logger
from defame.common.results import Result
from defame.tools.tool import Tool


class FaceRecognition(Action):
    name = "recognize_faces"
    description = "Identifies and recognizes faces within an image."
    how_to = "Provide an image and the model will recognize faces in it."
    format = "recognize_faces(<image:n>), where `n` is the image's ID"
    is_multimodal = True
    is_limited = True

    def __init__(self, image_ref: str):
        self.image: Image = MultimediaSnippet(image_ref).images[0]

    def __str__(self):
        return f'{self.name}({self.image.reference})'

    def __eq__(self, other):
        return isinstance(other, FaceRecognition) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


class FaceRecognizer(Tool):
    name = "face_recognizer"
    actions = [FaceRecognition]
    summarize = False

    def __init__(self, model_name: str = "face-recognition-model", **kwargs):
        super().__init__(**kwargs)
        # self.model = pipeline("image-classification", model=model_name, device=device)
        self.model = None

    def _perform(self, action: FaceRecognition) -> Result:
        return self.recognize_faces(action.image)

    def recognize_faces(self, image: torch.Tensor) -> Result:
        # TODO: Implement this method
        # results = self.model(image)
        # faces = [result['label'] for result in results]
        # return faces
        text = "Face Recognition is not implemented yet."
        logger.log(str(text))
        result = Result()
        return result

    def _summarize(self, result: Result, **kwargs) -> Optional[MultimediaSnippet]:
        raise NotImplementedError
