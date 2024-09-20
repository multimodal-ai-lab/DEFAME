from infact.common.logger import Logger
from infact.tools.credibility_checker import CredibilityChecker, CredibilityCheck
from .face_recognizer import FaceRecognizer, FaceRecognition
from .geolocator import Geolocator, Geolocate
from .manipulation_detector import ManipulationDetector, DetectManipulation
from .object_detector import ObjectDetector, DetectObjects
from .search.searcher import Searcher
from .search.common import WebSearch, WikiDumpLookup, WikiLookup, ReverseSearch, ImageSearch
from .text_extractor import TextExtractor, OCR
from .tool import Tool, get_available_actions

TOOL_REGISTRY = [
    CredibilityChecker,
    FaceRecognizer,
    Geolocator,
    ObjectDetector,
    Searcher,
    TextExtractor,
    ManipulationDetector,
]

ACTION_REGISTRY = {
    WebSearch,
    WikiDumpLookup,
    DetectObjects,
    WikiLookup,
    ReverseSearch,
    Geolocate,
    FaceRecognition,
    CredibilityCheck,
    OCR,
    DetectManipulation,
    ImageSearch,
}

IMAGE_ACTIONS = {
    ReverseSearch,
    Geolocate,
    FaceRecognition,
    OCR,
    DetectManipulation,
    DetectObjects,
    ImageSearch,
}


def get_tool_by_name(name: str):
    for t in TOOL_REGISTRY:
        if t.name == name:
            return t
    raise ValueError(f'Tool with name "{name}" does not exist.')


def initialize_tools(config: dict[str, dict], logger: Logger, device=None) -> list[Tool]:
    tools = []
    for tool_name, kwargs in config.items():
        tool_class = get_tool_by_name(tool_name)
        t = tool_class(**kwargs, logger=logger, device=device)
        tools.append(t)
    return tools
