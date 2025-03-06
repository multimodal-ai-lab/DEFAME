from defame.common import Report
from defame.evidence_retrieval.integrations.search_engines.common import WebSource
from defame.procedure.variants.qa_based.infact import InFact


class FirstResult(InFact):
    """InFact but using always the first result."""

    def answer_question(self,
                        question: str,
                        results: list[WebSource],
                        doc: Report = None) -> (str, WebSource):
        relevant_result = results[0]
        answer = self.attempt_answer_question(question, relevant_result, doc)
        return answer, relevant_result
