from typing import Any

from defame.common import Report, Label, logger
from defame.procedure.variants.summary_based.dynamic import DynamicSummary
from defame.evidence_retrieval.tools import Search


class AllActionsSummary(DynamicSummary):
    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        n_iterations = 0
        label = Label.NEI
        while label == Label.NEI and n_iterations < self.max_iterations:
            logger.log("Not enough information yet. Continuing fact-check...")
            n_iterations += 1
            actions, reasoning = self.planner.plan_next_actions(doc, all_actions=True)
            text = f'"{str(doc.claim).split(">", 1)[1].strip()}"'
            actions.append(Search(text))
            actions.append(Search(text, mode="images"))
            if len(reasoning) > 32:  # Only keep substantial reasoning
                doc.add_reasoning(reasoning)
            doc.add_actions(actions)
            if actions:
                evidences = self.actor.perform(actions, doc)
                doc.add_evidence(evidences)  # even if no evidence, add empty evidence block for the record
                self._develop(doc)
            label = self.judge.judge(doc, is_final=n_iterations == self.max_iterations or not actions)
        return label, {}
