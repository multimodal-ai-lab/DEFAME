from dataclasses import dataclass
from datetime import date as Date
from enum import Enum
from typing import Optional

from defame.common import Results, Image, MultimediaSnippet


class SearchMode(Enum):
    SEARCH = "search"
    NEWS = "news"
    PLACES = "places"
    IMAGES = "images"
    REVERSE = "reverse"  # Reverse Image Search (RIS)


@dataclass(frozen=True)
class Query:
    text: Optional[str] = None
    image: Optional[Image] = None
    limit: Optional[int] = None
    start_date: Optional[Date] = None
    end_date: Optional[Date] = None
    search_mode: Optional[SearchMode] = None

    def __post_init__(self):
        assert self.text or self.image, "Query must have at least one of 'text' or 'image'."

    def has_text(self) -> bool:
        return self.text is not None

    def has_image(self) -> bool:
        return self.image is not None

    def __eq__(self, other):
        return isinstance(other, Query) and self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((
            self.text,
            self.image,
            self.limit,
            self.start_date,
            self.end_date,
            self.search_mode,
        ))


class WebSource(MultimediaSnippet):
    """Output when searching the web or a local knowledge base."""

    def __init__(self,
                 *args,
                 url: str,
                 title: str = "",
                 date: Date = None,
                 summary: MultimediaSnippet = None,
                 query: Query = None,
                 rank: int = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.url = url
        self.title = title
        self.date = date
        self.summary = summary
        self.query = query
        self.rank = rank

    def is_relevant(self) -> Optional[bool]:
        """Returns true if the summary contains information helpful for the fact-check."""
        if self.summary is None:
            return None
        elif self.summary.data == "":
            return False
        else:
            return "NONE" not in self.summary.data

    def __str__(self):
        """Differentiates between direct citation (original text) and
        indirect citation (if summary is available)."""
        text = self.summary.data if self.summary is not None else f'"{self.data}"'
        return f'From [Source]({self.url}): {self.title}\nContent: {text}'

    def __eq__(self, other):
        return self.url == other.url

    def __hash__(self):
        return hash(self.url)


@dataclass
class SearchResults(Results):
    """A collection of web sources."""
    sources: list[WebSource]

    def __str__(self):
        return "**Web Search Result**\n\n" + "\n\n".join(map(str, self.sources))


@dataclass
class ReverseSearchResult(SearchResults):
    """Ships with additional object detection information next to the list of web sources."""
    entities: dict[str, float]  # mapping between entity description and confidence score
    best_guess_labels: list[str]

    def __str__(self):
        string_repr = "Google Vision's outputs"
        if self.entities:
            string_repr += f"\nIdentified web entities (confidence in parenthesis):\n"
            string_repr += "\n".join(f"{description} ({confidence * 100:.0f} %)"
                                     for description, confidence in self.entities.items())
        if self.best_guess_labels:
            string_repr += (f"\nBest guess about the topic of "
                            f"the image is {', '.join(self.best_guess_labels)}.\n Exact image matches found at:")
        return "**Reverse Search Result** The exact image was found in the following sources:\n\n" + "\n\n".join(
            map(str, self.sources))
