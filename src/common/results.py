from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List


class Result(ABC):
    """Detailed information piece retrieved by performing an action."""

    def is_useful(self) -> Optional[bool]:
        """Returns True if the contained information helps the fact-check."""
        raise NotImplementedError


@dataclass
class Evidence(ABC):
    """Information found during fact-checking. Is the output of performing
    an Action."""
    summary: str  # Key takeaways for the fact-check
    results: list[Result]

    def get_useful_results(self) -> list[Result]:
        return [r for r in self.results if r.is_useful()]


@dataclass
class SearchResult(Result):
    """Detailed information piece retrieved by performing an action."""
    source: str
    text: str
    date: datetime = None
    summary: str = None
    query: str = None
    rank: int = None

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


class GeolocationResult(Result):
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


class ObjectDetectionResult(Result):
    def __init__(self, source: str, objects: List[str], bounding_boxes: List[List[float]],
                 model_output: Optional[any] = None):
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
