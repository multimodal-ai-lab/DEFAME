from common.modeling import Model
from common.document import FCDoc
from eval.logger import EvaluationLogger
from safe.prompts.prompt import SummarizeDocPrompt


class DocSummarizer:
    """Summarizes a given, finished fact-checking document. The resulting summary is
    equivalent to the justification of the verdict."""
    def __init__(self, model: Model, logger: EvaluationLogger):
        self.model = model
        self.logger = logger

    def summarize(self, doc: FCDoc) -> str:
        summarize_doc_prompt = SummarizeDocPrompt(doc)
        summary = self.model.generate(str(summarize_doc_prompt))
        return summary
