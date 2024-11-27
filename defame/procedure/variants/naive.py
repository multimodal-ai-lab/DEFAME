from typing import Any

from defame.common import FCDocument, Label
from defame.procedure.procedure import Procedure


class NaiveQA(Procedure):
    """A naive approach where the claim veracity is
    predicted without any evidence."""

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        verdict = self.judge.judge_naively(doc)
        meta = dict(q_and_a=[])
        return verdict, meta
