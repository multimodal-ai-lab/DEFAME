from src.eval.logger import EvaluationLogger
from .credibility_checker import CredibilityChecker
from .face_recognizer import FaceRecognizer
from .geolocator import Geolocator
from .object_detector import ObjectDetector
from .search.searcher import Searcher
from .text_extractor import TextExtractor
from .tool import Tool, get_available_actions

TOOL_REGISTRY = [
    CredibilityChecker,
    FaceRecognizer,
    Geolocator,
    ObjectDetector,
    Searcher,
    TextExtractor,
]


def get_tool_by_name(name: str):
    for t in TOOL_REGISTRY:
        if t.name == name:
            return t
    raise ValueError(f'Tool with name "{name}" does not exist.')


def initialize_tools(config: dict[str, dict], logger: EvaluationLogger) -> list[Tool]:
    tools = []
    for tool_name, kwargs in config.items():
        tool_class = get_tool_by_name(tool_name)
        t = tool_class(**kwargs, logger=logger)
        tools.append(t)
    return tools
