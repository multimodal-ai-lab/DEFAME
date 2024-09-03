from typing import Any

from common import FCDocument, Label
from infact.procedure.procedure import Procedure


class NaiveQA(Procedure):
    """The naivest-possible approach where the claim veracity is
    predicted right away from the claim."""

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        verdict = self.judge.judge_naively(doc)
        meta = dict(q_and_a=[])
        return verdict, meta
