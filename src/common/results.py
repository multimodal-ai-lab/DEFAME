from abc import ABC
from typing import Optional
from datetime import datetime

from common.action import Action


class Result(ABC):
    """Evidence found during fact-checking."""
    source: str
    text: str
    date: datetime
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
