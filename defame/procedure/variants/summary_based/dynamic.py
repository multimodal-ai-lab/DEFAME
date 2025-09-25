from typing import Any

from defame.common import Report, Label, logger
from defame.procedure.procedure import Procedure


class DynamicSummary(Procedure):
    def __init__(self, max_iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        n_iterations = 0
        label = Label.NEI
        while label == Label.NEI and n_iterations < self.max_iterations:
            if n_iterations > 0:
                logger.log("Not enough information yet. Continuing fact-check...")
            n_iterations += 1
            actions, reasoning = self.planner.plan_next_actions(doc)
            if len(reasoning) > 32:  # Only keep substantial reasoning
                doc.add_reasoning(reasoning)
            if actions:
                doc.add_actions(actions)
                evidences = self.actor.perform(actions, doc)
                doc.add_evidence(evidences)  # even if no evidence, add empty evidence block for the record
                
                # Secondary planning: check if new evidence enables more specific actions
                secondary_actions, secondary_reasoning = self.planner.plan_next_actions(doc)
                if secondary_actions:
                    # Filter out actions we just completed, but allow URL-specific actions
                    completed_action_types = set(type(a) for a in actions)
                    
                    # Smart filtering: allow SearchX/SearchReddit even if SearchSocialX/SearchSocialReddit were performed
                    # Also allow multiple instances of the same URL-specific action type
                    new_actions = []
                    for action in secondary_actions:
                        action_type = type(action)
                        action_name = action_type.__name__
                        
                        # Always allow URL-specific social media actions (SearchX, SearchReddit)
                        if action_name in ['SearchX', 'SearchReddit']:
                            new_actions.append(action)
                        # For other actions, use the original filtering logic
                        elif action_type not in completed_action_types:
                            new_actions.append(action)
                    
                    if new_actions:
                        logger.info(f"Secondary planning found {len(new_actions)} additional actions")
                        if len(secondary_reasoning) > 32:
                            doc.add_reasoning("### Secondary Planning\n" + secondary_reasoning)
                        doc.add_actions(new_actions)
                        secondary_evidences = self.actor.perform(new_actions, doc)
                        doc.add_evidence(secondary_evidences)
                
                self._develop(doc)
            label = self.judge.judge(doc, is_final=n_iterations == self.max_iterations or not actions)
        return label, {}
