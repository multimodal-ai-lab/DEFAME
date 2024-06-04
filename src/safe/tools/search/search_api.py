from dataclasses import dataclass
from abc import ABC

from eval.logger import EvaluationLogger
from common.console import yellow
from typing import Optional


@dataclass
class SearchResult:
    url: str
    text: str
    query: str
    rank: int
    summary: str = None

    def __str__(self):
        """Human-friendly string representation in Markdown format.
        Differentiates between direct citation (original text) and
        indirect citation (if summary is available)."""
        text = self.summary or f'"{self.text}"'
        return f'{text} ([source]({self.url}))'

    def __eq__(self, other):
        return self.url == other.url

    def is_useful(self) -> Optional[bool]:
        """Returns true if the summary contains helpful information,
        e.g., does not contain NONE."""
        if self.summary is None:
            return None
        else:
            return "NONE" not in self.summary


class SearchAPI(ABC):
    """Abstract base class for all local and remote search APIs."""
    name: str
    is_free: bool
    is_local: bool

    def __init__(self, logger: EvaluationLogger = None):
        self.logger = logger
        self.total_searches = 0

    def _before_search(self, query: str):
        self.total_searches += 1
        if self.logger is not None:
            self.logger.log(yellow(f"Searching {self.name} with query: {query}"))

    def search(self, query: str, limit: int) -> list[SearchResult]:
        """Runs the API by submitting the query and obtaining a list of search results."""
        self._before_search(query)
        return self._call_api(query, limit)

    def _call_api(self, query: str, limit: int) -> list[SearchResult]:
        raise NotImplementedError()
