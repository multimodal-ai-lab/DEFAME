from src.tools.credibility_checker import CredibilityChecker
from src.eval.logger import EvaluationLogger
from src.tools.face_recognizer import FaceRecognizer
from src.tools.geolocator import Geolocator
from src.tools.object_detector import ObjectDetector
from src.tools.search.searcher import Searcher
from src.tools.text_extractor import TextExtractor
from src.tools.tool import Tool, get_available_actions

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


def initialize_tools(config: dict[str, dict], logger: EvaluationLogger, device=None) -> list[Tool]:
    tools = []
    for tool_name, kwargs in config.items():
        tool_class = get_tool_by_name(tool_name)
        t = tool_class(**kwargs, logger=logger, device=device)
        tools.append(t)
    return tools
