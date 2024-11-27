from typing import Any

from defame.common import Report, Label
from defame.procedure.procedure import Procedure


class Minimal(Procedure):
    """The most minimal approach where the claim veracity is
    predicted right away from the claim."""

    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        verdict = self.judge.judge_minimally(doc)
        meta = dict(q_and_a=[])
        return verdict, meta
