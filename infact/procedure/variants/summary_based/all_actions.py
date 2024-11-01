from typing import Any

from infact.common import FCDocument, Label
from infact.procedure.variants.summary_based.default import DynamicSummary


class AllActionsSummary(DynamicSummary):
    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        n_iterations = 0
        label = Label.NEI
        while label == Label.NEI and n_iterations < self.max_iterations:
            self.logger.log("Not enough information yet. Continuing fact-check...")
            n_iterations += 1
            actions, reasoning = self.planner.plan_next_actions(doc, all_actions=True)
            if len(reasoning) > 32:  # Only keep substantial reasoning
                doc.add_reasoning(reasoning)
            doc.add_actions(actions)
            if actions:
                evidences = self.actor.perform(actions, doc)
                doc.add_evidence(evidences)  # even if no evidence, add empty evidence block for the record
                self._develop(doc)
            label = self.judge.judge(doc, is_final=n_iterations == self.max_iterations or not actions)
        return label, {}
