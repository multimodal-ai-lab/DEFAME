from infact.common.action import CredibilityCheck
from infact.common.modeling import Model
from infact.common.results import Result
from infact.utils.parsing import is_url
from infact.tools.tool import Tool


class CredibilityChecker(Tool):
    """Evaluates the credibility of a given source (URL domain)."""
    name = "credibility_checker"
    actions = [CredibilityCheck]

    def __init__(self, llm: Model = None, **kwargs):
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
