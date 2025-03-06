from defame.common import Report, Action
from defame.evidence_retrieval.tools import WebSearch
from defame.procedure.variants.qa_based.infact import InFact


class NoQueryGeneration(InFact):
    """InFact but using the questions as search queries directly (instead of generating some)."""

    def propose_queries_for_question(self, question: str, doc: Report) -> list[Action]:
        return [WebSearch(f'"{question}"')]
