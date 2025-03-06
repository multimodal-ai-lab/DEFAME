from abc import ABC
from typing import Any

from defame.common import Report, Label, Model
from defame.common.misc import WebSource
from defame.modules import Judge, Actor, Planner
from defame.prompts.prompts import DevelopPrompt
from defame.evidence_retrieval.tools import WebSearch


class Procedure(ABC):
    """Base class of all procedures. A procedure is the algorithm which implements the fact-checking strategy."""

    def __init__(self, llm: Model, actor: Actor, judge: Judge, planner: Planner,
                 max_attempts: int = 3, **kwargs):
        self.llm = llm
        self.actor = actor
        self.judge = judge
        self.planner = planner
        self.max_attempts = max_attempts

    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        """Receives a fact-checking document (including a claim) and performs a fact-check on the claim.
        Returns the estimated veracity of the claim along with a dictionary, hosting any additional, procedure-
        specific meta information."""
        raise NotImplementedError

    def retrieve_resources(
            self,
            search_queries: list[WebSearch],
            doc: Report = None,
            summarize: bool = False
    ) -> list[WebSource]:
        search_results = []
        for query in search_queries:
            evidence = self.actor.perform([query], doc=doc, summarize=summarize)[0]
            if evidence.raw:
                search_results.extend(evidence.raw.sources)
        return search_results

    def _develop(self, doc: Report):
        """Analyzes the currently available information and infers new insights."""
        prompt = DevelopPrompt(doc)
        response = self.llm.generate(prompt)
        doc.add_reasoning("## Elaboration\n" + response)
