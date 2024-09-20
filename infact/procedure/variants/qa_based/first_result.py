from infact.common import FCDocument
from infact.common.misc import WebSource
from infact.procedure.variants.qa_based.infact import InFact


class FirstResult(InFact):
    """InFact but using always the first result."""

    def answer_question(self,
                        question: str,
                        results: list[WebSource],
                        doc: FCDocument = None) -> (str, WebSource):
        relevant_result = results[0]
        answer = self.attempt_answer_question(question, relevant_result, doc)
        return answer, relevant_result
