from common.action import *
from common.modeling import Model
from common.results import Result, GeoLocationResult, OCRResult, ObjectDetectionResult
from eval.logger import EvaluationLogger
from safe.searcher import Searcher

from typing import Optional, List, Dict
from transformers import pipeline, AutoProcessor, AutoModelForObjectDetection, AutoModel
import torch
from PIL import Image
import numpy as np
import easyocr


class ObjectDetector:
    def __init__(self, model_name: str = "facebook/detr-resnet-50", device: int = -1, use_multiple_gpus: bool = False):
        """
        Initialize the ObjectDetector with a pretrained model from Hugging Face.

        :param model_name: The name of the Hugging Face model to use for object detection.
        :param device: The device to run the model on (e.g., -1 for CPU, 0 for GPU).
        :param use_multiple_gpus: Whether to use multiple GPUs if available.
        """
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != -1 else "cpu")

        if use_multiple_gpus and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

    def recognize_objects(self, image: Image.Image) -> ObjectDetectionResult:
        """
        Recognize objects in an image.

        :param image: A PIL image.
        :return: An ObjectDetectionResult instance containing recognized objects and their bounding boxes.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        objects = [self.model.module.config.id2label[label.item()] if hasattr(self.model, 'module') else self.model.config.id2label[label.item()] for label in results["labels"]]
        bounding_boxes = [box.tolist() for box in results["boxes"]]

        return ObjectDetectionResult(source=self.model_name, objects=objects, bounding_boxes=bounding_boxes, model_output=outputs)

# Example usage:
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# object_detector = ObjectDetector(use_multiple_gpus=True)
# objects = object_detector.recognize_objects(image)
# print(objects)


class GeoLocator:
    def __init__(self, model_name: str = "geolocal/StreetCLIP", top_k = 10, device: int = -1, use_multiple_gpus: bool = False):
        """
        Initialize the GeoLocator with a pretrained model from Hugging Face.
        
        :param model_name: The name of the Hugging Face model to use for geolocation.
        :param device: The device to run the model on (e.g., -1 for CPU, 0 for GPU).
        :param use_multiple_gpus: Whether to use multiple GPUs if available.
        """
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != -1 else "cpu")
        self.top_k = top_k

        if use_multiple_gpus and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

    def locate(self, image: Image.Image, choices: List[str] = None) -> GeoLocationResult:
        """
        Perform geolocation on an image.
        
        :param image: A PIL image.
        :param choices: A list of location choices. If None, uses a default list of countries.
        :return: A GeoLocationResult object containing location predictions and their probabilities.
        """
        if choices is None:
            choices = ['Albania', 'Andorra', 'Argentina', 'Australia', 'Austria', 'Bangladesh', 'Belgium', 'Bermuda',
                       'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'China',
                       'Colombia', 'Croatia', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ecuador', 'Estonia',
                       'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Greenland', 'Guam', 'Guatemala', 'Hungary',
                       'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kenya',
                       'Kyrgyzstan', 'Laos', 'Latvia', 'Lesotho', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar',
                       'Malaysia', 'Malta', 'Mexico', 'Monaco', 'Mongolia', 'Montenegro', 'Netherlands', 'New Zealand',
                       'Nigeria', 'Norway', 'Pakistan', 'Palestine', 'Peru', 'Philippines', 'Poland', 'Portugal',
                       'Puerto Rico', 'Romania', 'Russia', 'Rwanda', 'Senegal', 'Serbia', 'Singapore', 'Slovakia',
                       'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Swaziland', 'Sweden',
                       'Switzerland', 'Taiwan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Arab Emirates',
                       'United Kingdom', 'United States', 'Uruguay']

        inputs = self.processor(text=choices, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        prediction = logits_per_image.softmax(dim=1)

        # Compute classification score for each country
        confidences = {choices[i]: float(prediction[0][i].item()) for i in range(len(choices))}
        top_k_locations = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:self.top_k])
        most_likely_location = max(top_k_locations, key=top_k_locations.get)
        model_output = logits_per_image
        return GeoLocationResult(
            source=self.model_name,
            text=f"The most likely countries where the image was taken are: {top_k_locations}",
            most_likely_location=most_likely_location,
            top_k_locations=top_k_locations,
            model_output=model_output
        )

# Example usage:
# url = "https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg"
# image = Image.open(requests.get(url, stream=True).raw)
# geo_locator = GeoLocator(use_multiple_gpus=True)
# top_locations = geo_locator.locate(image)
# print(top_locations)
    

class FaceRecognizer:
    def __init__(self, model_name: str = "face-recognition-model", device: int = -1):
        #self.model = pipeline("image-classification", model=model_name, device=device)
        self.model = None

    def recognize_faces(self, image: torch.Tensor) -> Result:
        #results = self.model(image)
        #faces = [result['label'] for result in results]
        #return faces
        result = Result()
        result.text = "Face Recognition is not implemented yet."
        result.source = self.model
        return result
    
#TODO: create an acutal SourceCredibilityChecker
class SourceCredibilityChecker:
    def __init__(self, model: Model = None):
        self.model = model

    def check_credibility(self, source: List[str]) -> List[Result]:
        #credibility_score = self.model(source)
        #return credibility_score
        result = Result()
        result.text = "Source Credibility Check is not implemented yet."
        result.source = self.model
        return [result]
    
#TODO: Integrate a differentiable OCR Reader. Potentially open-source like PaddleOCR
#class OCR:
#    def __init__(self, model_name: str = "ocr-model", device: int = -1):
#        self.model = pipeline("image-to-text", model=model_name, device=device)
#
#    def extract_text(self, image: torch.Tensor) -> str:
#        results = self.model(image)
#        text = results[0]['generated_text']
#        return text


class OCR:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the OCR tool with EasyOCR.
        
        :param use_gpu: Whether to use GPU for OCR.
        """
        self.model = None #TODO: Later we could have a trainable OCR model here
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Perform OCR on an image.
        
        :param image: A PIL image.
        :return: An OCRResult object containing the extracted text.
        """
        results = self.reader.readtext(np.array(image))
        # Concatenate all detected text pieces
        extracted_text = ' '.join([result[1] for result in results])
        return OCRResult(source="EasyOCR", text=extracted_text, model_output=results)


# Example usage:
# url = "https://example.com/some_image.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# ocr_tool = OCR(use_gpu=True)
# text = ocr_tool.extract_text(image)
# print(text)

class Actor:
    def __init__(self,
                 multimodal: bool,
                 model: str | Model,
                 object_detector: str | ObjectDetector = None,
                 reverse_searcher: str | Searcher = None,
                 geo_locator: str | GeoLocator = None,
                 face_recognizer: str | FaceRecognizer = None,
                 source_checker: str | SourceCredibilityChecker = None,
                 image_reader: str | OCR = None,
                 search_engines: list[str] = None,
                 max_results_per_search: int = 3,
                 logger: EvaluationLogger = None,
                 ):

        self.logger = logger or EvaluationLogger()
        if multimodal:
            self.object_detector = ObjectDetector(model_name=object_detector) if isinstance(object_detector, str) else ObjectDetector()
            self.geo_locator = GeoLocator(model_name=geo_locator) if isinstance(geo_locator, str) else GeoLocator(top_k = 10)
            #TODO: implement a correct reverse searcher.
            self.reverse_searcher = reverse_searcher or Searcher(search_engines=["google"],
                                                                 model=model, 
                                                                 logger=self.logger, 
                                                                 summarize=False, 
                                                                 limit_per_search=max_results_per_search)
            self.face_recognizer = FaceRecognizer(model_name=face_recognizer) if isinstance(face_recognizer, str) else FaceRecognizer()
            self.image_reader = OCR(model_name=image_reader) if isinstance(image_reader, str) else OCR()
        #TODO: implement source_checker
        self.source_checker = SourceCredibilityChecker(model_name=source_checker) if isinstance(face_recognizer, str) else SourceCredibilityChecker()
        search_engines = search_engines or ["duckduckgo"]
        self.searcher = Searcher(search_engines, model, self.logger,
                                 summarize=False,
                                 limit_per_search=max_results_per_search)

    def perform(self, actions: list[Action], images: Optional[list[Image.Image]] = None) -> list[Result]:
        # TODO: Enable parallelization here, e.g. through async calls
        all_results = []
        for a in actions:
            all_results.extend(self._perform_single(a))
        return all_results

    def _perform_single(self, action: Action) -> list[Result]:
        if isinstance(action, Search):
            return self._perform_search(action)
        elif isinstance(action, ObjectRecognition):
                return self._perform_object_recognition(action)
        elif isinstance(action, ReverseSearch):
                return(self._perform_reverse_search(action))
        elif isinstance(action, GeoLocation):
                return self._perform_geo_location(action)
        elif isinstance(action, FaceRecognition):
                return self._perform_face_recognition(action)
        elif isinstance(action, SourceCredibilityCheck):
                return self._perform_source_credibility_check(action)
        elif isinstance(action, OCR):
                return self._perform_ocr(action)
        else:
            raise ValueError(f"Action '{action}' unknown.")

    def _perform_search(self, search: Search) -> list[Result]:
        return self.searcher.search(search.query)  # TODO: split into different searchers
    
    def _perform_object_recognition(self, action: ObjectRecognition) -> ObjectDetectionResult:
        #TODO: The Result can contain more detailed info. probably the actor needs to have a multimodal model and a general model
        self.logger.log(f"Performing object recognition on the image.")
        return self.object_detector.recognize_objects(action.image)

    def _perform_reverse_search(self, action: ReverseSearch) -> list[Result]:
        self.logger.log(f"Performing reverse image search.")
        return self.reverse_searcher.search(action.image)
    
    def _perform_geo_location(self, action: GeoLocation) -> GeoLocationResult:
        self.logger.log(f"Performing geolocation on the image.")
        return self.geo_locator.locate(action.image)
    
    def _perform_face_recognition(self, action: FaceRecognition) -> Result:
        self.logger.log(f"Performing face recognition on the image.")
        return self.face_recognizer.recognize_faces(action.image)
    
    def _perform_source_credibility_check(self, action: SourceCredibilityCheck) -> list[Result]:
        self.logger.log(f"Performing source credibility check for {action.source}.")
        return self.source_checker.check_credibility(action.source)
    
    def _perform_ocr(self, action: OCR) -> OCRResult:
        self.logger.log(f"Performing OCR on the image.")
        return self.image_reader.extract_text(action.image)
    

