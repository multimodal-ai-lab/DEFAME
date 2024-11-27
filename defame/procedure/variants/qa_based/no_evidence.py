from defame.common import Report
from defame.procedure.variants.qa_based.infact import InFact
from defame.prompts.prompts import AnswerQuestionNoEvidence


class NoEvidence(InFact):
    """InFact but without any evidence retrieval."""

    def approach_question_batch(self, questions: list[str], doc: Report) -> list:
        q_and_a = []
        doc.add_reasoning("## Research Q&A")
        for question in questions:
            prompt = AnswerQuestionNoEvidence(question, doc)
            response = self.llm.generate(prompt)
            qa_string = (f"### {question}\n"
                         f"Answer: {response}")
            doc.add_reasoning(qa_string)
            qa_instance = {
                "question": question,
                "answer": response,
                "url": "",
                "scraped_text": "",
            }
            q_and_a.append(qa_instance)
        return q_and_a
