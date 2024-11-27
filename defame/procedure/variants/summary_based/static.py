from typing import Any

from defame.common import Report, Label
from defame.procedure.procedure import Procedure


class StaticSummary(Procedure):
    def __init__(self, max_iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        actions, reasoning = self.planner.plan_next_actions(doc)
        if len(reasoning) > 32:  # Only keep substantial reasoning
            doc.add_reasoning(reasoning)
        doc.add_actions(actions)
        if actions:
            evidences = self.actor.perform(actions, doc)
            doc.add_evidence(evidences)  # even if no evidence, add empty evidence block for the record
            self._develop(doc)
        label = self.judge.judge(doc, is_final=True)
        return label, {}
