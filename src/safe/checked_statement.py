
import dataclasses
from typing import Any

import safe.fact_checker
from safe import rate_atomic_fact

IRRELEVANT_LABEL = 'Irrelevant'
SUPPORTED_LABEL = safe.fact_checker.SUPPORTED_LABEL
NOT_SUPPORTED_LABEL = safe.fact_checker.NOT_SUPPORTED_LABEL

_MAX_PIPELINE_RETRIES = 3


class CheckedStatement:
    """Class for storing checked statements."""

    def __init__(
            self,
            sentence: str,
            atomic_fact: str,
            self_contained_atomic_fact: str,
            relevance_data: dict[str, Any] | None = None,
            rate_data: safe.fact_checker.FinalAnswer | None = None,
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
