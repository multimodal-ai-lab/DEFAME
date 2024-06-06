from abc import ABC
from typing import Optional

from common.action import Action


class Result(ABC):
    """Evidence found during fact-checking."""
    source: str
    text: str
    summary: str
    action: Action

    def is_useful(self) -> Optional[bool]:
        """Returns true if the summary contains helpful information,
        e.g., does not contain NONE."""
        if self.summary is None:
            return None
        else:
            return "NONE" not in self.summary

    def __str__(self):
        """Human-friendly string representation in Markdown format.
        Differentiates between direct citation (original text) and
        indirect citation (if summary is available)."""
        text = self.summary or f'"{self.text}"'
        return f'{text} ([source]({self.source}))'

    def __eq__(self, other):
        return self.source == other.source

    def __hash__(self):
        return hash(self.source)


class SearchResult(Result):
    query: str
    rank: int

    def __init__(self, url: str, text: str, query: str,
                 rank: int = None, action: Action = None):
        self.source = url
        self.text = text
        self.query = query
        self.rank = rank
        self.action = action
