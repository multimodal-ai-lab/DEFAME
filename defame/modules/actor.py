from datetime import datetime
from typing import Optional, List

from defame.common import Action, Report, Evidence
from defame.evidence_retrieval.tools import Tool, Searcher


class Actor:
    """Agent that executes given Actions and returns the resulted Evidence."""

    def __init__(self, tools: list[Tool]):
        self.tools = tools

    def perform(self, actions: list[Action], doc: Report = None, summarize: bool = True) -> list[Evidence]:
        # TODO: Parallelize
        all_evidence = []
        social_media_actions = []
        regular_actions = []
        
        # Separate social media actions from regular actions
        for action in actions:
            assert isinstance(action, Action)
            action_name = type(action).__name__
            # Only aggregate dedicated social media tools (SearchReddit, SearchX) that have actual content
            # Don't aggregate searcher social actions (SearchSocialReddit, SearchSocialX) that only find URLs
            if action_name in ['SearchX', 'SearchReddit']:
                social_media_actions.append(action)
            else:
                regular_actions.append(action)
        
        # Perform regular actions normally
        for action in regular_actions:
            all_evidence.append(self._perform_single(action, doc, summarize=summarize))
        
        # Aggregate social media actions if any exist
        if social_media_actions:
            aggregated_evidence = self._perform_social_media_aggregation(social_media_actions, doc, summarize=summarize)
            all_evidence.append(aggregated_evidence)
        
        return all_evidence

    def _perform_single(self, action: Action, doc: Report = None, summarize: bool = True) -> Evidence:
        tool = self.get_corresponding_tool_for_action(action)
        return tool.perform(action, summarize=summarize, doc=doc)

    def _perform_social_media_aggregation(self, social_media_actions: list[Action], doc: Report = None, summarize: bool = True) -> Evidence:
        """Aggregate all social media actions into a single evidence item."""
        from defame.common.evidence import Evidence
        from defame.common.modeling import make_model
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Perform each social media action individually to get raw results
        reddit_results = []
        x_results = []
        
        for action in social_media_actions:
            try:
                evidence = self._perform_single(action, doc, summarize=False)  # Get raw results first
                action_name = type(action).__name__
                
                if action_name == 'SearchReddit':
                    reddit_results.append(evidence)
                elif action_name == 'SearchX':
                    x_results.append(evidence)
            except Exception as e:
                logger.warning(f"Failed to process social media action {action}: {e}")
                continue
        
        # Create aggregated summary using LLM
        if reddit_results or x_results:
            return self._create_aggregated_social_media_evidence(reddit_results + x_results, doc)
        else:
            # Fallback: return empty evidence if all actions failed
            from defame.common.content import MultimodalSequence
            
            # Create a simple action object for the aggregation
            class FailedSocialMediaAction:
                name = "FailedSocialMedia"
            
            # Create a simple Results-like object
            class SimpleResults:
                def __init__(self, content):
                    self.content = content
                def __str__(self):
                    return self.content
            
            failed_action = FailedSocialMediaAction()
            failed_results = SimpleResults("No social media evidence could be retrieved.")
            
            return Evidence(
                raw=failed_results,
                action=failed_action,
                takeaways=MultimodalSequence(["Social media analysis failed."])
            )

    def _create_aggregated_social_media_evidence(self, social_media_evidence: List[Evidence], doc) -> Evidence:
        """Create a single aggregated evidence from multiple social media evidence items using bulk LLM analysis."""
        import logging
        from defame.common.content import MultimodalSequence
        from defame.prompts.prompts import AnalyzeBulkSocialMediaPrompt
        
        logger = logging.getLogger(__name__)
        
        # Collect raw content and credibility from all social media evidence
        all_raw_content = []
        source_actions = []
        
        for i, evidence in enumerate(social_media_evidence, 1):
            if evidence.raw:
                # Get the raw social media content with credibility assessment
                raw_result = evidence.raw
                action_name = evidence.action.name
                
                # Extract credibility score if available
                credibility_info = ""
                if hasattr(raw_result, 'credibility_score'):
                    credibility_info = f" (Credibility: {raw_result.credibility_score})"
                
                # Extract the full content from the WebSource, not the truncated __str__ representation
                full_content = ""
                if hasattr(raw_result, 'content') and raw_result.content:
                    # This is a RedditResults/XResults object with a WebSource in content attribute
                    web_source = raw_result.content
                    if hasattr(web_source, 'content') and web_source.content:
                        # Get the actual MultimodalSequence content from the WebSource
                        full_content = str(web_source.content)
                        logger.info(f"📄 Extracted MultimodalSequence content for {action_name}: {len(full_content)} characters")
                    else:
                        # WebSource doesn't have content, use its string representation  
                        full_content = str(web_source)
                        logger.info(f"📄 Using WebSource string for {action_name}: {len(full_content)} characters")
                else:
                    # Fallback to string representation of the raw result
                    full_content = str(raw_result)
                    logger.info(f"📄 Fallback content for {action_name}: {len(full_content)} characters")
                
                # Format the raw content for bulk analysis
                source_content = f"=== SOURCE {i} ({action_name}){credibility_info} ===\n"
                source_content += f"{action_name} Content from {getattr(raw_result, 'url', 'unknown URL')}:\n"
                source_content += full_content
                all_raw_content.append(source_content)
                source_actions.append(action_name)
        
        if not all_raw_content:
            # If no content was found, return a default evidence
            class AggregateAction:
                name = "AggregatedSocialMedia"
            
            class SimpleResults:
                def __init__(self, content):
                    self.content = content
                def __str__(self):
                    return self.content
            
            dummy_action = AggregateAction()
            dummy_results = SimpleResults("No social media content available")
            
            return Evidence(
                raw=dummy_results,
                action=dummy_action,
                takeaways=MultimodalSequence(["No useful social media evidence was found."])
            )
        
        # Combine all raw content for bulk analysis
        combined_sources = "\n\n".join(all_raw_content)
        
        # Log what we're sending for aggregation
        logger.info("📊 SOCIAL MEDIA AGGREGATION INPUT:")
        logger.info(f"Total sources: {len(all_raw_content)}")
        logger.info(f"Source actions: {source_actions}")
        logger.info(f"Combined content length: {len(combined_sources)} characters")
        logger.info("Combined content preview:")
        logger.info(combined_sources[:1000] + ("..." if len(combined_sources) > 1000 else ""))
        
        # Use LLM to analyze all social media content together
        try:
            # Get LLM from one of the tools that has it
            llm = None
            for tool in self.tools:
                if hasattr(tool, 'llm') and tool.llm is not None:
                    llm = tool.llm
                    break
            
            if llm is None:
                raise Exception("No LLM available in any tool")
                
            bulk_prompt = AnalyzeBulkSocialMediaPrompt(combined_sources, doc)
            
            # Log the full prompt being sent to LLM
            logger.info("🔍 BULK SOCIAL MEDIA ANALYSIS PROMPT:")
            logger.info("=" * 80)
            logger.info(str(bulk_prompt))
            logger.info("=" * 80)
            
            takeaways = llm.generate(bulk_prompt, max_attempts=3)
            
            # Log the LLM response
            logger.info("🤖 LLM RESPONSE FOR BULK ANALYSIS:")
            logger.info("-" * 80)
            logger.info(f"Response length: {len(takeaways) if takeaways else 0} characters")
            logger.info(f"Response content: {takeaways}")
            logger.info("-" * 80)
            
            if not takeaways or takeaways.strip().upper() == "NONE":
                takeaways = "No useful social media evidence was found across all sources."
                
        except Exception as e:
            logger.warning(f"Failed to generate bulk social media analysis: {e}")
            takeaways = f"Analysis of {len(all_raw_content)} social media sources ({', '.join(set(source_actions))}) completed, but detailed extraction failed."
        
        # Create aggregated evidence
        class AggregatedSocialMediaAction:
            name = "AggregatedSocialMedia"
        
        class SimpleResults:
            def __init__(self, content):
                self.content = content
            def __str__(self):
                return self.content
        
        aggregated_action = AggregatedSocialMediaAction()
        # Store the raw combined content for debugging/reference
        aggregated_results = SimpleResults(combined_sources)
        
        return Evidence(
            raw=aggregated_results,
            action=aggregated_action,
            takeaways=MultimodalSequence([takeaways])
        )

    def get_corresponding_tool_for_action(self, action: Action) -> Tool:
        for tool in self.tools:
            if type(action) in tool.actions:
                return tool
        raise ValueError(f"No corresponding tool available for Action '{action}'.")

    def reset(self):
        """Resets all tools (if applicable)."""
        for tool in self.tools:
            tool.reset()

    def set_current_claim_id(self, claim_id: str):
        for tool in self.tools:
            tool.set_claim_id(claim_id)

    def get_tool_stats(self):
        return {t.name: t.get_stats() for t in self.tools}

    def _get_searcher(self) -> Optional[Searcher]:
        for tool in self.tools:
            if isinstance(tool, Searcher):
                return tool

    def set_search_date_restriction(self, before: Optional[datetime]):
        searcher = self._get_searcher()
        if searcher is not None:
            searcher.set_time_restriction(before)
