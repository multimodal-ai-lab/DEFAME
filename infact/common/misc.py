from dataclasses import dataclass
from datetime import date as Date
from typing import Optional

from infact.common import MultimediaSnippet


@dataclass
class Query:
    text: str
    limit: int = None
    start_date: Date = None
    end_date: Date = None

    def __eq__(self, other):
        return (isinstance(other, Query) and
                self.text == other.text and
                self.start_date == other.start_date and
                self.end_date == other.end_date)

    def __hash__(self):
        return hash(self.text + str(self.start_date) + str(self.end_date))


@dataclass
class WebSource(MultimediaSnippet):
    """Output when searching the web or a local knowledge base."""
    url: str
    date: Date = None
    summary: MultimediaSnippet = None
    query: Query = None
    rank: int = None

    def is_useful(self) -> Optional[bool]:
        """Returns true if the summary contains helpful information,
        e.g., does not contain NONE."""
        if self.summary is None:
            return None
        elif self.summary.text == "":
            return False
        else:
            return "NONE" not in self.summary.text

    def __str__(self):
        """Differentiates between direct citation (original text) and
        indirect citation (if summary is available)."""
        text = self.summary.text if self.summary is not None else f'"{self.text}"'
        return f'From [Source]({self.url}):\n{text}'

    def __eq__(self, other):
        return self.url == other.url

    def __hash__(self):
        return hash(self.url)
