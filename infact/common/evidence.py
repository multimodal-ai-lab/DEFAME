from dataclasses import dataclass
from typing import Optional

from infact.common.results import Result
from infact.common.action import Action
from infact.common.medium import MultimediaSnippet


@dataclass
class Evidence:
    """Any chunk of possibly helpful information found during the
    fact-check. Is typically the output of performing an Action."""
    raw: Result  # The raw output from the executed tool
    action: Action  # The action which led to this evidence
    summary: MultimediaSnippet = None  # Contains the key takeaways for the fact-check

    def is_useful(self) -> Optional[bool]:
        """Returns True if the contained information helps the fact-check."""
        return self.summary is not None

    def __str__(self):
        if self.is_useful():
            return str(self.summary)
        else:
            return str(self.raw)

    # def get_useful_results(self) -> list[Result]:
    #     return [r for r in self.raw if r.is_useful()]
