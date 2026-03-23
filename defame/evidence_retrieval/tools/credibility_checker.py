from typing import Optional

from ezmm import MultimodalSequence

from defame.common import Action, logger, Model, Results
from defame.utils.parsing import is_url
from defame.evidence_retrieval.tools.tool import Tool


class CredibilityCheck(Action):
    """Evaluates the credibility of a given source."""
    name = "check_credibility"

    def __init__(self, source: str):
        """
        @param source: The URL or name of the source to determine the credibility of.
        """
        self._save_parameters(locals())
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

    def _perform(self, action: CredibilityCheck) -> Results:
        return self.check_credibility(action.source)

    def check_credibility(self, source: str) -> Results:
        # TODO: Actually implement this method
        # credibility_score = self.model(source)
        # return credibility_score
        text = "Source Credibility Check is not implemented yet. "
        if is_url(source):
            text += f'{source} is a valid url.'
        logger.log(str(text))
        result = Results()
        return result

    def _summarize(self, result: Results, **kwargs) -> Optional[MultimodalSequence]:
        raise NotImplementedError
