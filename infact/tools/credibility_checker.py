from typing import Optional

from infact.common import MultimediaSnippet, Action, logger, Model, Result
from infact.utils.parsing import is_url
from infact.tools.tool import Tool


class CredibilityCheck(Action):
    name = "check_credibility"
    description = "Evaluates the credibility of a given source."
    how_to = "Provide a source URL or name and the model will assess its credibility."
    format = 'check_credibility("url")'
    is_multimodal = False
    is_limited = False

    def __init__(self, source: str):
        self.source = source

    def __str__(self):
        return f'{self.name}("{self.source}")'

    def __eq__(self, other):
        return isinstance(other, CredibilityCheck) and self.source == other.source

    def __hash__(self):
        return hash((self.name, self.source))


class CredibilityChecker(Tool):
    """Evaluates the credibility of a given source (URL domain)."""
    name = "credibility_checker"
    actions = [CredibilityCheck]
    summarize = False

    def __init__(self, llm: Model = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    def _perform(self, action: CredibilityCheck) -> Result:
        return self.check_credibility(action.source)

    def check_credibility(self, source: str) -> Result:
        # TODO: Actually implement this method
        # credibility_score = self.model(source)
        # return credibility_score
        text = "Source Credibility Check is not implemented yet. "
        if is_url(source):
            text += f'{source} is a valid url.'
        logger.log(str(text))
        result = Result()
        return result

    def _summarize(self, result: Result, **kwargs) -> Optional[MultimediaSnippet]:
        raise NotImplementedError
