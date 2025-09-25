from typing import Optional

from defame.common.modeling import Model
from defame.evidence_retrieval.tools.credibility_checker import CredibilityChecker, CredibilityCheck
from .face_recognizer import FaceRecognizer, FaceRecognition
from .geolocator import Geolocator, Geolocate
from .manipulation_detector import ManipulationDetector, DetectManipulation
from .object_detector import ObjectDetector, DetectObjects
from .reddit import RedditTool, SearchReddit, RedditResults
from .searcher import Searcher, Search
from .text_extractor import TextExtractor, OCR
from .tool import Tool
from .x import XTool, SearchX, XResults

TOOL_REGISTRY = [
    CredibilityChecker,
    FaceRecognizer,
    Geolocator,
    ObjectDetector,
    RedditTool,
    Searcher,
    TextExtractor,
    ManipulationDetector,
    XTool,
]

ACTION_REGISTRY = {
    Search,
    DetectObjects,
    Geolocate,
    FaceRecognition,
    CredibilityCheck,
    OCR,
    DetectManipulation,
    SearchReddit,
    SearchX,
}

# Import the new social media search actions
from defame.evidence_retrieval.tools.searcher import SearchSocialReddit, SearchSocialX

# Add them to the registry
ACTION_REGISTRY.update({
    SearchSocialReddit,
    SearchSocialX,
})

IMAGE_ACTIONS = {
    Geolocate,
    FaceRecognition,
    OCR,
    DetectManipulation,
    DetectObjects,
}


def get_tool_by_name(name: str):
    for t in TOOL_REGISTRY:
        if t.name == name:
            return t
    raise ValueError(f'Tool with name "{name}" does not exist.')


def initialize_tools(config: dict[str, dict], llm: Optional[Model], device=None) -> list[Tool]:
    tools = []
    if config:
        for tool_name, kwargs in config.items():
            if kwargs is None:
                kwargs = {}
            kwargs.update({"llm": llm, "device": device})
            tool_class = get_tool_by_name(tool_name)
            t = tool_class(**kwargs)
            tools.append(t)
    return tools
