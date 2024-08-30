from infact.common import FCDocument, Action
from infact.common.action import WebSearch
from infact.procedure.variants.qa_based.infact import InFact


class NoQueryGeneration(InFact):
    """InFact but using the questions as search queries directly (instead of generating some)."""

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[Action]:
        return [WebSearch(f'"{question}"')]
