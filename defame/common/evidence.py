from dataclasses import dataclass
from typing import Optional

from ezmm import MultimodalSequence

from defame.common.results import Results
from defame.common.action import Action


@dataclass
class Evidence:
    """Any chunk of possibly helpful information found during the
    fact-check. Is typically the output of performing an Action."""
    raw: Results  # The raw output from the executed tool
    action: Action  # The action which led to this evidence
    takeaways: Optional[MultimodalSequence]  # Contains all info helpful for the fact-check, if any

    def is_useful(self) -> bool:
        """Returns True if the contained information helps the fact-check,
        i.e., if there are any takeaways."""
        return self.takeaways is not None

    def __str__(self):
        header = f"### Evidence from `{self.action.name}`\n"
        body = str(self.takeaways if self.takeaways else self.raw)
        return header + body
