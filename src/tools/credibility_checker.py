from src.common.action import CredibilityCheck
from src.common.modeling import LLM
from src.common.results import Result
from src.utils.parsing import is_url
from src.eval.logger import EvaluationLogger
from src.tools.tool import Tool


class CredibilityChecker(Tool):
    """Evaluates the credibility of a given source (URL domain)."""
    name = "credibility_checker"
    actions = [CredibilityCheck]

    def __init__(self, llm: LLM = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    def perform(self, action: CredibilityCheck) -> list[Result]:
        return [self.check_credibility(action.source)]

    def check_credibility(self, source: str) -> Result:
        # TODO: Actually implement this method
        # credibility_score = self.model(source)
        # return credibility_score
        text = "Source Credibility Check is not implemented yet. "
        if is_url(source):
            text += f'{source} is a valid url.'
        self.logger.log(str(text))
        result = Result()
        return result
