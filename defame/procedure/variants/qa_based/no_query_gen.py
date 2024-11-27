from defame.common import FCDocument, Action
from defame.tools import WebSearch
from defame.procedure.variants.qa_based.infact import InFact


class NoQueryGeneration(InFact):
    """InFact but using the questions as search queries directly (instead of generating some)."""

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[Action]:
        return [WebSearch(f'"{question}"')]
