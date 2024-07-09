from common.document import FCDocument
from common.modeling import LLM
from eval.logger import EvaluationLogger
from prompts.prompt import SummarizeDocPrompt


class DocSummarizer:
    """Summarizes a given, finished fact-checking document. The resulting summary is
    equivalent to the justification of the verdict."""

    def __init__(self, llm: LLM, logger: EvaluationLogger):
        self.llm = llm
        self.logger = logger

    def summarize(self, doc: FCDocument) -> str:
        summarize_doc_prompt = SummarizeDocPrompt(doc)
        summary = self.llm.generate(str(summarize_doc_prompt))
        return summary
