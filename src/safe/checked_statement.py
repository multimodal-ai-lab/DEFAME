import dataclasses
from typing import Any

from safe.judge import FinalAnswer


class CheckedStatement:
    """Class for storing checked statements. Currently unused."""

    def __init__(
            self,
            sentence: str,
            atomic_fact: str,
            self_contained_atomic_fact: str,
            relevance_data: dict[str, Any] | None = None,
            rate_data: FinalAnswer | None = None,
            annotation: str = '',
    ):
        self.sentence = sentence
        self.atomic_fact = atomic_fact
        self.self_contained_atomic_fact = self_contained_atomic_fact
        self.relevance_data = relevance_data
        self.rate_data = rate_data
        self.annotation = annotation
        self.data = {
            'sentence': self.sentence,
            'atomic_fact': self.atomic_fact,
            'self_contained_atomic_fact': self.self_contained_atomic_fact,
            'relevance_data': self.relevance_data if self.relevance_data else None,
            'rate_data': (
                dataclasses.asdict(self.rate_data) if self.rate_data else None
            ),
            'annotation': self.annotation,
        }
