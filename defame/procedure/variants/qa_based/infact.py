from typing import Any

from defame.common import Report, Label
from defame.procedure.variants.qa_based.base import QABased


class InFact(QABased):
    """The procedure as implemented by InFact, using all six stages (stage 6, justification
    generation, follows outside of this method)."""

    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        # Stage 1 & 2: Interpretation & Question posing
        questions = self._pose_questions(no_of_questions=10, doc=doc)

        # Stages 3 & 4: Search query generation and question answering
        q_and_a = self.approach_question_batch(questions, doc)

        # Stage 5: Veracity prediction
        label = self.judge.judge(doc)

        return label, dict(q_and_a=q_and_a)
