from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date as Date
from typing import Optional

from defame.common import MultimediaSnippet, Image, Medium


@dataclass(frozen=True)
class Query(ABC):
    limit: Optional[int] = None
    start_date: Optional[Date] = None
    end_date: Optional[Date] = None
    search_type: str = 'search'

    @abstractmethod
    def get_query_content(self):
        """Abstract method to retrieve the main query content."""
        pass

    def __eq__(self, other):
        if not isinstance(other, Query):
            return False
        return (
                self.limit == other.limit and
                self.start_date == other.start_date and
                self.end_date == other.end_date and
                self.search_type == other.search_type
        )

    def __hash__(self):
        return hash((
            self.limit,
            self.start_date,
            self.end_date,
            self.search_type
        ))


@dataclass(frozen=True)
class TextQuery(Query):
    text: str = ''
    search_type: str = 'search'

    def get_query_content(self):
        return self.text

    def __eq__(self, other):
        if not isinstance(other, TextQuery):
            return False
        return super().__eq__(other) and self.text == other.text

    def __hash__(self):
        return hash((super().__hash__(), self.text))


@dataclass(frozen=True)
class ImageQuery(Query):
    text: str = None
    image: Image = None
    search_type: str = 'image'

    def get_query_content(self):
        return self.image

    def __eq__(self, other):
        if not isinstance(other, ImageQuery):
            return False
        return super().__eq__(other) and self.image == other.image

    def __hash__(self):
        return hash((super().__hash__(), self.image))


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
