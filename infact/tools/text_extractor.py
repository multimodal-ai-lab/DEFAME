# import easyocr
from PIL import Image
import numpy as np

from infact.common.action import OCR
from infact.common.results import OCRResult, Result
from infact.tools.tool import Tool


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
    summarize = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the OCR tool with EasyOCR.

        :param use_gpu: Whether to use GPU for OCR.
        """
        self.model = None  # TODO: Later we could have a trainable OCR model here
        # self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def perform(self, action: OCR) -> list[Result]:
        return [self.extract_text(action.image.image)]

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
