# import easyocr
from dataclasses import dataclass, field
from typing import Optional

from PIL.Image import Image as PILImage
import numpy as np

from defame.common import MultimediaSnippet, Action, Image, logger
from defame.common.results import Results
from defame.evidence_retrieval.tools.tool import Tool


# TODO: Integrate a differentiable OCR Reader. Potentially open-source like PaddleOCR
# class OCR:
#    def __init__(self, model_name: str = "ocr-model", device: int = -1):
#        self.model = pipeline("image-to-text", model=model_name, device=device)
#
#    def extract_text(self, image: torch.Tensor) -> str:
#        results = self.model(image)
#        text = results[0]['generated_text']
#        return text


class OCR(Action):
    name = "ocr"
    description = "Performs Optical Character Recognition to extract text from an image."
    how_to = "Provide an image and the model will extract text from it."
    format = "ocr(<image:n>), where `n` is the image's ID"
    is_multimodal = True

    def __init__(self, image_ref: str):
        self.image: Image = MultimediaSnippet(image_ref).images[0]

    def __str__(self):
        return f'{self.name}({self.image.reference})'

    def __eq__(self, other):
        return isinstance(other, OCR) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


@dataclass
class OCRResults(Results):
    source: str
    extracted_text: str
    model_output: Optional[any] = None
    text: str = field(init=False)  # This will be assigned in __post_init__

    def __post_init__(self):
        self.text = str(self)

    def __str__(self):
        return f'From [Source]({self.source}):\nExtracted Text: {self.extracted_text}'

    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None


class TextExtractor(Tool):
    """Employs OCR to get all the text visible in the image."""
    name = "text_extractor"
    actions = [OCR]
    summarize = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the OCR tool with EasyOCR.

        :param use_gpu: Whether to use GPU for OCR.
        """
        self.model = None  # TODO: Later we could have a trainable OCR model here
        # self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def _perform(self, action: OCR) -> Results:
        return self.extract_text(action.image.image)

    def extract_text(self, image: PILImage) -> OCRResults:
        """
        Perform OCR on an image.

        :param image: A PIL image.
        :return: An OCRResult object containing the extracted text.
        """
        results = self.reader.readtext(np.array(image))
        # Concatenate all detected text pieces
        extracted_text = ' '.join([result[1] for result in results])
        result = OCRResults(source="EasyOCR", text=extracted_text, model_output=results)
        logger.log(str(result))
        return result

    def _summarize(self, result: OCRResults, **kwargs) -> Optional[MultimediaSnippet]:
        # TODO: Add image reference, summarize the output w.r.t. relevant content
        return MultimediaSnippet(f"Extracted text: {result.text}")
