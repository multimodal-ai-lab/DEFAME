from tools.credibility_checker import CredibilityChecker
from eval.logger import EvaluationLogger
from tools.face_recognizer import FaceRecognizer
from tools.geolocator import Geolocator
from tools.object_detector import ObjectDetector
from tools.search.searcher import Searcher
from tools.text_extractor import TextExtractor
from tools.tool import Tool, get_available_actions

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
