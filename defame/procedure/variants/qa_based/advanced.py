from typing import Any

from defame.common import Report, Label
from defame.evidence_retrieval.integrations.search.common import Source
from defame.procedure.variants.qa_based.base import QABased
from defame.prompts.prompts import AnswerCollectively


class AdvancedQA(QABased):
    """The former "dynamic" or "multi iteration" approach. Intended as improvement over
    InFact but turned out to have worse performance on AVeriTeC."""

    def __init__(self, max_iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        # Run iterative Q&A as long as there is NEI
        q_and_a = []
        n_iterations = 0
        label = Label.REFUSED_TO_ANSWER
        while n_iterations < self.max_iterations:
            n_iterations += 1

            questions = self._pose_questions(no_of_questions=4, doc=doc)
            new_qa_instances = self.approach_question_batch(questions, doc)
            q_and_a.extend(new_qa_instances)

            if (label := self.judge.judge(doc)) != Label.NEI:
                break

        # Fill up QA with more questions
        missing_questions = 10 - len(q_and_a)
        if missing_questions > 0:
            questions = self._pose_questions(no_of_questions=missing_questions, doc=doc)
            new_qa_instances = self.approach_question_batch(questions, doc)
            q_and_a.extend(new_qa_instances)

        return label, dict(q_and_a=q_and_a)

    def answer_question(self,
                        question: str,
                        sources: list[Source],
                        doc: Report = None) -> (str, Source):
        """Generates an answer to the given question by considering batches of 5 search results at once."""
        for i in range(0, len(sources), 5):
            sources_batch = sources[i:i + 5]
            prompt = AnswerCollectively(question, sources_batch, doc)
            out = self.llm.generate(prompt, max_attempts=3)
            if out is not None:
                if out["answered"]:
                    answer = out["answer"]
                    result_id = out["result_id"]
                    result = sources_batch[result_id]
                    return answer, result

        # No search result helpful to answer the question
        return None, None
