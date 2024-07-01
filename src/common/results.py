from abc import ABC
from typing import Optional, List
from datetime import datetime

from common.action import Action


class Result(ABC):
    """Evidence found during fact-checking."""
    source: str
    text: str
    date: datetime = None
    summary: str = None

    def is_useful(self) -> Optional[bool]:
        """Returns true if the summary contains helpful information,
        e.g., does not contain NONE."""
        if self.summary is None:
            return None
        elif self.summary == "":
            return False
        else:
            return "NONE" not in self.summary

    def __str__(self):
        """Human-friendly string representation in Markdown format.
        Differentiates between direct citation (original text) and
        indirect citation (if summary is available)."""
        text = self.summary or f'"{self.text}"'
        return f'From [Source]({self.source}):\n{text}'

    def __eq__(self, other):
        return self.source == other.source

    def __hash__(self):
        return hash(self.source)


class SearchResult(Result):
    query: str
    rank: int

    def __init__(self, url: str, text: str, date: datetime, query: str,
                 rank: int = None, action: Action = None):
        self.source = url
        self.text = text
        self.date = date
        self.query = query
        self.rank = rank
        self.action = action

class GeoLocationResult(Result):
    def __init__(self, source: str, text: str, most_likely_location: str,
                 top_k_locations: List[str], model_output: Optional[any] = None):
        self.text = text
        self.source = source
        self.most_likely_location = most_likely_location
        self.top_k_locations = top_k_locations
        self.model_output = model_output

    def __str__(self):
        """Human-friendly string representation in Markdown format."""
        locations_str = ', '.join(self.top_k_locations)
        text = f'Most likely location: {self.most_likely_location}\nTop {len(self.top_k_locations)} locations: {locations_str}'
        return f'From [Source]({self.source}):\n{text}'
    
    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None

class OCRResult(Result):
    def __init__(self, source: str, text: str, model_output: Optional[any] = None):
        self.text = text
        self.source = source
        self.model_output = model_output

    def __str__(self):
        """Human-friendly string representation in Markdown format."""
        return f'From [Source]({self.source}):\nExtracted Text: {self.text}'
    
    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None
    
class ObjectDetectionResult:
    def __init__(self, source: str, objects: List[str], bounding_boxes: List[List[float]], model_output: Optional[any] = None):
        self.source = source
        self.objects = objects
        self.bounding_boxes = bounding_boxes
        self.model_output = model_output

    def __str__(self):
        """Human-friendly string representation in Markdown format."""
        objects_str = ', '.join(self.objects)
        boxes_str = ', '.join([str(box) for box in self.bounding_boxes])
        return f'From [Source]({self.source}):\nObjects: {objects_str}\nBounding boxes: {boxes_str}'
    
    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None
    
    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None

