from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL.Image import Image as PILImage
from transformers import AutoProcessor, AutoModelForObjectDetection

from infact.common import Result, MultimediaSnippet, Action, Image
from infact.tools.tool import Tool


class DetectObjects(Action):
    name = "detect_objects"
    description = "Identifies objects within an image."
    how_to = "Provide an image and the model will recognize objects in it."
    format = "detect_objects(<image:n>), where `n` is the image's ID"
    is_multimodal = True
    is_limited = True

    def __init__(self, image_ref: str):
        self.image: Image = MultimediaSnippet(image_ref).images[0]

    def __str__(self):
        return f'{self.name}({self.image.reference})'

    def __eq__(self, other):
        return isinstance(other, DetectObjects) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


@dataclass
class ObjectDetectionResult(Result):
    source: str
    objects: List[str]
    bounding_boxes: List[List[float]]
    model_output: Optional[any] = None
    text: str = field(init=False)  # This will be assigned in __post_init__

    def __post_init__(self):
        # After initialization, generate the text field using the string representation
        self.text = str(self)

    def __str__(self):
        objects_str = ', '.join(self.objects)
        boxes_str = ', '.join([str(box) for box in self.bounding_boxes])
        return f'From [Source]({self.source}):\nObjects: {objects_str}\nBounding boxes: {boxes_str}'

    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None


class ObjectDetector(Tool):
    name = "object_detector"
    actions = [DetectObjects]
    summarize = False

    def __init__(self, model_name: str = "facebook/detr-resnet-50", **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the ObjectDetector with a pretrained model from Hugging Face.

        :param model_name: The name of the Hugging Face model to use for object detection.
        :param device: The device to run the model on (e.g., -1 for CPU, 0 for GPU).
        :param use_multiple_gpus: Whether to use multiple GPUs if available.
        """
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.device = torch.device(self.device if self.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

    def _perform(self, action: DetectObjects) -> ObjectDetectionResult:
        return self.recognize_objects(action.image.image)

    def recognize_objects(self, image: PILImage) -> ObjectDetectionResult:
        """
        Recognize objects in an image.

        :param image: A PIL image.
        :return: An ObjectDetectionResult instance containing recognized objects and their bounding boxes.
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            objects = [self.model.module.config.id2label[label.item()] if hasattr(self.model, 'module') else
                       self.model.config.id2label[label.item()] for label in results["labels"]]
            bounding_boxes = [box.tolist() for box in results["boxes"]]

        result = ObjectDetectionResult(
            source=self.model_name,
            objects=objects,
            bounding_boxes=bounding_boxes,
            model_output=outputs)

        self.logger.log(str(result))
        return result

    def _summarize(self, result: ObjectDetectionResult, **kwargs) -> Optional[MultimediaSnippet]:
        return MultimediaSnippet("Object Detector not fully implemented yet.")  # TODO: Implement the summary
