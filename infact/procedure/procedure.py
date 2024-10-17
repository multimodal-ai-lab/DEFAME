from abc import ABC
from typing import Any

from infact.tools import WebSearch
from infact.common.misc import WebSource
from infact.common import FCDocument, Label, Model, Logger
from infact.modules import Judge, Actor, Planner


class Procedure(ABC):
    """Base class of all procedures. A procedure is the algorithm which implements the fact-checking strategy."""

    def __init__(self, llm: Model, actor: Actor, judge: Judge, planner: Planner, logger: Logger,
                 max_attempts: int = 3, **kwargs):
        self.llm = llm
        self.actor = actor
        self.judge = judge
        self.planner = planner
        self.logger = logger
        self.max_attempts = max_attempts

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        """Receives a fact-checking document (including a claim) and performs a fact-check on the claim.
        Returns the estimated veracity of the claim along with a dictionary, hosting any additional, procedure-
        specific meta information."""
        raise NotImplementedError

    def retrieve_resources(
            self,
            search_queries: list[WebSearch],
            doc: FCDocument = None,
            summarize: bool = False
    ) -> list[WebSource]:
        search_results = []
        for query in search_queries:
            evidence = self.actor.perform([query], doc=doc, summarize=summarize)[0]
            if evidence.raw:
                search_results.extend(evidence.raw.sources)
        return search_results
