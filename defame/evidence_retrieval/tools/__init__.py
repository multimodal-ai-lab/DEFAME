from typing import Optional

from defame.common.modeling import Model
from defame.evidence_retrieval.tools.credibility_checker import CredibilityChecker, CredibilityCheck
from .face_recognizer import FaceRecognizer, FaceRecognition
from .geolocator import Geolocator, Geolocate
from .manipulation_detector import ManipulationDetector, DetectManipulation
from .object_detector import ObjectDetector, DetectObjects
from .searcher import Searcher, Search
from .text_extractor import TextExtractor, OCR
from .tool import Tool

# Import social media tools
try:
    from .x_tool import XTool, SearchX
    from .reddit_tool import RedditTool, SearchReddit
    SOCIAL_MEDIA_TOOLS_AVAILABLE = True
except ImportError:
    SOCIAL_MEDIA_TOOLS_AVAILABLE = False
    XTool = None
    RedditTool = None
    SearchX = None
    SearchReddit = None

TOOL_REGISTRY = [
    CredibilityChecker,
    FaceRecognizer,
    Geolocator,
    ObjectDetector,
    Searcher,
    TextExtractor,
    ManipulationDetector,
]

# Add social media tools if available
if SOCIAL_MEDIA_TOOLS_AVAILABLE:
    TOOL_REGISTRY.extend([XTool, RedditTool])

ACTION_REGISTRY = {
    Search,
    DetectObjects,
    Geolocate,
    FaceRecognition,
    CredibilityCheck,
    OCR,
    DetectManipulation,
}

# Add social media actions if available
if SOCIAL_MEDIA_TOOLS_AVAILABLE:
    ACTION_REGISTRY.update({SearchX, SearchReddit})

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
