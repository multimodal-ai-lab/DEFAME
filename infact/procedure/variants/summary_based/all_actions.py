from typing import Any, Collection

from infact.common import FCDocument, Label, Evidence
from infact.procedure.procedure import Procedure
from infact.prompts.prompts import ReiteratePrompt
from infact.tools.search.common import WebSearch, ImageSearch


class AllActionsSummary(Procedure):
    def __init__(self, max_iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        n_iterations = 0
        label = Label.NEI
        while label == Label.NEI and n_iterations < self.max_iterations:
            self.logger.log("Not enough information yet. Continuing fact-check...")
            n_iterations += 1
            actions, reasoning = self.planner.plan_next_actions(doc, all_actions=True)
            text = f'"{doc.claim.text.split(">", 1)[1].strip()}"'
            actions.append(WebSearch(text))
            actions.append(ImageSearch(text))
            if len(reasoning) > 32:  # Only keep substantial reasoning
                doc.add_reasoning(reasoning)
            doc.add_actions(actions)
            if actions:
                evidences = self.actor.perform(actions, doc)
                doc.add_evidence(evidences)  # even if no evidence, add empty evidence block for the record
                self._consolidate_knowledge(doc, evidences)
            label = self.judge.judge(doc, is_final=n_iterations == self.max_iterations or not actions)
        return label, {}

    def _consolidate_knowledge(self, doc: FCDocument, evidences: Collection[Evidence]):
        """Analyzes the currently available information and states new questions, adds them
        to the FCDoc."""
        prompt = ReiteratePrompt(doc, evidences)
        response = self.llm.generate(prompt)
        doc.add_reasoning(response)
