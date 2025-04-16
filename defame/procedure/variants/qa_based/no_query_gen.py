from defame.common import Report, Action
from defame.evidence_retrieval import Search
from defame.procedure.variants.qa_based.infact import InFact


class NoQueryGeneration(InFact):
    """InFact but using the questions as search queries directly (instead of generating some)."""

    def propose_queries_for_question(self, question: str, doc: Report) -> list[Action]:
        return [Search(f'"{question}"')]
