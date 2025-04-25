from typing import Optional

import torch
from ezmm import MultimodalSequence, Image

from defame.common import Action, logger
from defame.common.results import Results
from defame.evidence_retrieval.tools.tool import Tool


class FaceRecognition(Action):
    """Identifies and recognizes faces within an image."""
    name = "recognize_faces"

    requires_image = True

    def __init__(self, image: str):
        """
        @param image: The reference of the image to recognize faces in.
        """
        self._save_parameters(locals())
        self.image = Image(reference=image)

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

    def _perform(self, action: FaceRecognition) -> Results:
        return self.recognize_faces(action.image)

    def recognize_faces(self, image: torch.Tensor) -> Results:
        # TODO: Implement this method
        # results = self.model(image)
        # faces = [result['label'] for result in results]
        # return faces
        text = "Face Recognition is not implemented yet."
        logger.log(str(text))
        result = Results()
        return result

    def _summarize(self, result: Results, **kwargs) -> Optional[MultimodalSequence]:
        raise NotImplementedError
