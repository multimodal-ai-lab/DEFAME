from typing import Any

from defame.common import Report, Label, logger
from .dynamic import DynamicSummary


class NoDevelop(DynamicSummary):
    """Like Dynamic but without develop"""
    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        n_iterations = 0
        label = Label.NEI
        while label == Label.NEI and n_iterations < self.max_iterations:
            logger.log("Not enough information yet. Continuing fact-check...")
            n_iterations += 1
            actions, reasoning = self.planner.plan_next_actions(doc)
            if len(reasoning) > 32:  # Only keep substantial reasoning
                doc.add_reasoning(reasoning)
            if actions:
                doc.add_actions(actions)
                evidences = self.actor.perform(actions, doc)
                doc.add_evidence(evidences)  # even if no evidence, add empty evidence block for the record
                # self._develop(doc)
            label = self.judge.judge(doc, is_final=n_iterations == self.max_iterations or not actions)
        return label, {}
