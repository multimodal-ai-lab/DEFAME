from defame.common import Report
from defame.evidence_retrieval.integrations.search.common import Source
from defame.procedure.variants.qa_based.infact import InFact


class FirstResult(InFact):
    """InFact but using always the first result."""

    def answer_question(self,
                        question: str,
                        sources: list[Source],
                        doc: Report = None) -> (str, Source):
        relevant_source = sources[0]
        answer = self.attempt_answer_question(question, relevant_source, doc)
        return answer, relevant_source
