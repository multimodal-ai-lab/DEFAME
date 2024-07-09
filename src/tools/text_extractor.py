# import easyocr
from PIL import Image
import numpy as np

from common.action import OCR
from common.results import OCRResult, Result
from eval.logger import EvaluationLogger
from tools.tool import Tool


# TODO: Integrate a differentiable OCR Reader. Potentially open-source like PaddleOCR
# class OCR:
#    def __init__(self, model_name: str = "ocr-model", device: int = -1):
#        self.model = pipeline("image-to-text", model=model_name, device=device)
#
#    def extract_text(self, image: torch.Tensor) -> str:
#        results = self.model(image)
#        text = results[0]['generated_text']
#        return text


class TextExtractor(Tool):
    """Employs OCR to get all the text visible in the image."""
    name = "text_extractor"
    actions = [OCR]

    def __init__(self, use_gpu: bool = True, logger: EvaluationLogger = None):
        """
        Initialize the OCR tool with EasyOCR.

        :param use_gpu: Whether to use GPU for OCR.
        """
        self.model = None  # TODO: Later we could have a trainable OCR model here
        # self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        self.logger = logger

    def perform(self, action: OCR) -> list[Result]:
        return [self.extract_text(action.image)]

    def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Perform OCR on an image.

        :param image: A PIL image.
        :return: An OCRResult object containing the extracted text.
        """
        results = self.reader.readtext(np.array(image))
        # Concatenate all detected text pieces
        extracted_text = ' '.join([result[1] for result in results])
        result = OCRResult(source="EasyOCR", text=extracted_text, model_output=results)
        self.logger.log(str(result))
        return result
