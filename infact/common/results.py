from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import numpy as np

from infact.common.medium import MultimediaSnippet, Image



class Result(MultimediaSnippet, ABC):
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
    url: str
    text: str
    date: datetime = None
    summary: str = None
    query: str = None
    image: Image = None
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
        return f'From [Source]({self.url}):\n{text}'

    def __eq__(self, other):
        return self.url == other.url

    def __hash__(self):
        return hash(self.url)


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


@dataclass
class ObjectDetectionResult(Result):
    source: str
    objects: List[str]
    bounding_boxes: List[List[float]]
    model_output: Optional[any] = None
    text: str = field(init=False)  # This will be assigned in __post_init__

    def __post_init__(self):
        # After initialization, generate the text field using the string representation
        self.text = str(self)

    def __str__(self):
        """Human-friendly string representation in Markdown format."""
        objects_str = ', '.join(self.objects)
        boxes_str = ', '.join([str(box) for box in self.bounding_boxes])
        return f'From [Source]({self.source}):\nObjects: {objects_str}\nBounding boxes: {boxes_str}'

    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None


@dataclass
class OCRResult(Result):
    source: str
    extracted_text: str 
    model_output: Optional[any] = None
    text: str = field(init=False)  # This will be assigned in __post_init__

    def __post_init__(self):
        self.text = str(self)

    def __str__(self):
        """Human-friendly string representation in Markdown format."""
        return f'From [Source]({self.source}):\nExtracted Text: {self.extracted_text}'

    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None
    
@dataclass
class ManipulationResult(Result):
    text: str = field(init=False)
    score: Optional[float]
    confidence_map: Optional[np.ndarray]
    localization_map: np.ndarray
    ref_confidence_map: Optional[str]
    ref_localization_map: Optional[str]
    noiseprint: Optional[np.ndarray] = None
    

    def is_useful(self) -> Optional[bool]:
        return self.score is not None or self.confidence_map is not None

    def __str__(self):
        score_str = f'Score: {self.score:.3f}' if self.score is not None else 'Score: N/A'
        conf_str = f'Confidence map available: {self.ref_confidence_map}' if self.confidence_map is not None else 'Confidence map: N/A'
        loc_str = f'Localization map available: {self.ref_localization_map}' if self.localization_map is not None else 'Localization map: N/A'
        noiseprint_str = 'Noiseprint++ available' if self.noiseprint is not None else 'Noiseprint++: N/A'

        return f'Manipulation Detection Resultswi\n{score_str}\n{conf_str}\n{loc_str}\n{noiseprint_str}'
    
    def __post_init__(self):
        self.text = str(self)


