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
                
                # Post-process: aggregate social media evidence if multiple sources exist
                evidences = self._aggregate_social_media_if_needed(evidences, doc)
                
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
                        
                        # Post-process: aggregate social media evidence if multiple sources exist
                        secondary_evidences = self._aggregate_social_media_if_needed(secondary_evidences, doc)
                        
                        doc.add_evidence(secondary_evidences)
                
                self._develop(doc)
            label = self.judge.judge(doc, is_final=n_iterations == self.max_iterations or not actions)
        return label, {}
    
    def _aggregate_social_media_if_needed(self, evidences: list, doc) -> list:
        """
        Check if there are multiple social media evidence items and aggregate them if needed.
        This implements bulk social media analysis at the procedure level.
        """
        # Import here to avoid circular imports
        from defame.evidence_retrieval.tools.reddit import SearchReddit
        from defame.evidence_retrieval.tools.x import SearchX
        
        # Find social media evidence in the current batch
        social_media_evidence = []
        other_evidence = []
        
        for evidence in evidences:
            if evidence.action and isinstance(evidence.action, (SearchReddit, SearchX)):
                social_media_evidence.append(evidence)
            else:
                other_evidence.append(evidence)
        
        logger.info(f"🔍 PROCEDURE: Found {len(social_media_evidence)} social media evidence, {len(other_evidence)} other evidence")
        
        # If we have multiple social media evidence items, aggregate them
        if len(social_media_evidence) >= 2:
            # Get the SocialMediaAggregator tool
            aggregator = self._get_social_media_aggregator()
            
            if aggregator:
                logger.info(f"📊 PROCEDURE: Aggregating {len(social_media_evidence)} social media sources")
                aggregated_evidence = aggregator.aggregate_social_media_evidence(social_media_evidence, doc)
                
                # Return other evidence + aggregated evidence (replace individual social media evidence)
                return other_evidence + [aggregated_evidence]
            else:
                logger.warning("⚠️ SocialMediaAggregator not found - keeping individual evidence")
        
        # No aggregation needed or no aggregator available
        return evidences
    
    def _get_social_media_aggregator(self):
        """Get the SocialMediaAggregator tool from the actor's tools."""
        from defame.evidence_retrieval.tools.social_media_aggregator import SocialMediaAggregator
        
        for tool in self.actor.tools:
            if isinstance(tool, SocialMediaAggregator):
                return tool
        return None
