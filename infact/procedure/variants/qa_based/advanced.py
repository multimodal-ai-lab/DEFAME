from typing import Any, Optional

from common import FCDocument, Label, SearchResult
from infact.procedure.variants.qa_based.base import QABased
from infact.prompts.prompt import AnswerCollectively
from utils.parsing import extract_last_code_span, extract_last_paragraph


class AdvancedQA(QABased):
    """The former "dynamic" or "multi iteration" approach. Intended as improvement over
    InFact but turned out to have worse performance on AVeriTeC."""
    def __init__(self, max_iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
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
                        results: list[SearchResult],
                        doc: FCDocument = None) -> (str, SearchResult):
        """Generates an answer to the given question by considering batches of 5 search results at once."""
        for i in range(0, len(results), 5):
            results_batch = results[i:i + 5]
            prompt = AnswerCollectively(question, results_batch, doc)
            response = self.llm.generate(prompt, max_attempts=3)

            # Extract result ID and answer to the question from response
            if "NONE" not in response and "None" not in response:
                try:
                    result_id = extract_last_code_span(response)
                    if result_id != "":
                        result_id = int(result_id)
                        answer = extract_last_paragraph(response)
                        return answer, results_batch[result_id]
                except:
                    pass
        return None, None
