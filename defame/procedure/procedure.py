from abc import ABC
from typing import Any

from defame.common import Report, Label, Model
from defame.evidence_retrieval import Source, Search
from defame.evidence_retrieval.integrations import SearchResults
from defame.modules import Judge, Actor, Planner
from defame.prompts.prompts import DevelopPrompt


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

    def retrieve_sources(
            self,
            search_actions: list[Search],
            doc: Report = None,
            summarize: bool = False
    ) -> list[Source]:
        evidence = self.actor.perform(search_actions, doc=doc, summarize=summarize)
        sources = []
        for e in evidence:
            results = e.raw
            if isinstance(results, SearchResults):
                sources.extend(results.sources)
        return sources

    def _develop(self, doc: Report):
        """Analyzes the currently available information and infers new insights."""
        prompt = DevelopPrompt(doc)
        response = self.llm.generate(prompt)
        doc.add_reasoning("## Elaboration\n" + response)
