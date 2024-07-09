from common.action import CredibilityCheck
from common.modeling import LLM
from common.results import Result
from utils.parsing import is_url
from eval.logger import EvaluationLogger
from tools.tool import Tool


class CredibilityChecker(Tool):
    """Evaluates the credibility of a given source (URL domain)."""
    name = "credibility_checker"
    actions = [CredibilityCheck]

    def __init__(self, llm: LLM = None, logger: EvaluationLogger = None):
        self.llm = llm
        self.logger = logger

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
