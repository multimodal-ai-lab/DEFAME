from infact.procedure.variants.qa_based.infact import InFact
from common.document import FCDocument
from utils.parsing import find_code_span
from prompts.prompt import PoseQuestionsPrompt


class NoInterpretation(InFact):
    """InFact but without interpretation."""

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        """Generates some questions that needs to be answered during the fact-check."""
        prompt = PoseQuestionsPrompt(doc, n_questions=no_of_questions, interpret=False)
        response = self.llm.generate(prompt)
        # Extract the questions
        questions = find_code_span(response)
        return questions
