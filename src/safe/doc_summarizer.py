from common.modeling import Model
from common.document import FCDoc
from eval.logger import EvaluationLogger


class DocSummarizer:
    """Summarizes a given, finished fact-checking document. The resulting summary is
    equivalent to the justification of the verdict."""
    def __init__(self, model: Model, logger: EvaluationLogger):
        self.model = model
        self.logger = logger

    def summarize(self, doc: FCDoc) -> str:
        raise NotImplementedError()
