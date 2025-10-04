"""
Social Media Aggregator Tool for combining multiple social media evidence sources.
"""

import logging
from typing import List, Optional, Dict, Any

from defame.common import Action, Evidence, Report
from defame.common.content import MultimodalSequence
from defame.evidence_retrieval.tools.tool import Tool


class AggregateSocialMedia(Action):
    """Action for aggregating social media evidence from multiple sources."""
    name = "aggregate_social_media"
    
    def __init__(self, social_media_actions: List[Action]):
        """
        @param social_media_actions: List of social media actions to aggregate
        """
        self._save_parameters(locals())
        self.social_media_actions = social_media_actions

    def __eq__(self, other):
        return isinstance(other, AggregateSocialMedia) and self.social_media_actions == other.social_media_actions

    def __hash__(self):
        return hash((self.name, tuple(self.social_media_actions)))


class SocialMediaAggregator(Tool):
    """Tool for aggregating multiple social media evidence sources into a single coherent evidence item."""
    
    name = "social_media_aggregator"
    description = "A tool for aggregating multiple social media evidence sources into a single coherent evidence item."
    
    # Class-level registry for social media tools
    _social_media_tools: Dict[str, 'Tool'] = {}
    
    def __init__(self, llm=None, device=None, **kwargs):
        super().__init__(llm, device)
        self.actions = [AggregateSocialMedia]
        self.logger = logging.getLogger(__name__)
        
    @classmethod
    def register_social_media_tool(cls, tool_instance: 'Tool', action_types: List[type]):
        """Register a social media tool that can be aggregated."""
        for action_type in action_types:
            cls._social_media_tools[action_type.__name__] = tool_instance
            
    @classmethod 
    def is_social_media_action(cls, action: Action) -> bool:
        """Check if an action is a social media action that should be aggregated."""
        return type(action).__name__ in cls._social_media_tools
        
    def perform(self, action: Action, summarize: bool = True, **kwargs) -> Evidence:
        """Perform aggregation of social media actions."""
        if isinstance(action, AggregateSocialMedia):
            return self._perform_aggregation(action, **kwargs)
        else:
            raise ValueError(f"SocialMediaAggregator cannot handle action type: {type(action)}")
            
    def _perform_aggregation(self, action: AggregateSocialMedia, **kwargs) -> Evidence:
        """Execute social media aggregation."""
        doc = kwargs.get('doc')
        
        # Execute each social media action using its registered tool
        social_media_evidence = []
        
        for sm_action in action.social_media_actions:
            tool = self._social_media_tools.get(type(sm_action).__name__)
            if tool:
                try:
                    evidence = tool.perform(sm_action, summarize=False, **kwargs)
                    social_media_evidence.append(evidence)
                except Exception as e:
                    self.logger.warning(f"Failed to process social media action {sm_action}: {e}")
                    continue
        
        # Aggregate the evidence
        return self.aggregate_social_media_evidence(social_media_evidence, doc)

    def aggregate_social_media_evidence(self, social_media_evidence: List[Evidence], doc: Report = None) -> Evidence:
        """
        Main method to aggregate multiple social media evidence into a single coherent evidence.
        This method provides bulk analysis of all social media sources to reduce LLM usage.
        
        @param social_media_evidence: List of Evidence objects from social media sources
        @param doc: Optional Report object to use for context
        @return: Single Evidence object containing aggregated analysis
        """
        if not social_media_evidence:
            return self._create_empty_evidence("No social media evidence provided for aggregation")
            
        source_names = [e.action.name for e in social_media_evidence]
        self.logger.info(f"📊 Starting bulk social media aggregation for {len(social_media_evidence)} sources: {source_names}")
        
        result = self._create_aggregated_evidence(social_media_evidence, doc)
        
        # Calculate actual content length from MultimodalSequence
        takeaways_content = ""
        if result.takeaways and hasattr(result.takeaways, 'content'):
            # MultimodalSequence has a content attribute that contains the actual text
            takeaways_content = " ".join([str(item) for item in result.takeaways.content if item])
        elif result.takeaways:
            takeaways_content = str(result.takeaways)
        
        content_length = len(takeaways_content.strip())
        self.logger.info(f"✅ Completed bulk social media aggregation - generated {content_length} characters of analysis")
        
        return result

    def can_aggregate(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """
        Generic interface method for Actor to check if this tool can aggregate any evidence.
        Returns a list of evidence that this tool can aggregate.
        """
        # Define social media tool action names that this aggregator can handle
        social_media_tools = {'search_x', 'search_reddit', 'search_facebook', 'search_instagram', 'search_tiktok'}
        
        # Find evidence from social media tools
        aggregatable_evidence = [e for e in evidence_list if e.action.name.lower() in social_media_tools]
        
        return aggregatable_evidence
    
    def aggregate_evidence(self, evidence_list: List[Evidence], doc: Report = None) -> Evidence:
        """
        Generic interface method for Actor to trigger aggregation.
        Simply delegates to the existing aggregate_social_media_evidence method.
        """
        return self.aggregate_social_media_evidence(evidence_list, doc)

    def _create_aggregated_evidence(self, social_media_evidence: List[Evidence], doc: Report) -> Evidence:
        """Create a single aggregated evidence from multiple social media evidence items using bulk LLM analysis."""
        from defame.prompts.prompts import AnalyzeBulkSocialMediaPrompt
        
        # Collect raw content and credibility from all social media evidence
        all_raw_content = []
        source_actions = []
        
        for i, evidence in enumerate(social_media_evidence, 1):
            if evidence.raw:
                raw_result = evidence.raw
                action_name = evidence.action.name
                
                # Extract credibility score if available
                credibility_info = ""
                if hasattr(raw_result, 'credibility_score'):
                    credibility_info = f" (Credibility: {raw_result.credibility_score})"
                
                # Extract the full content from the WebSource
                full_content = self._extract_content_from_result(raw_result, action_name)
                
                # Format the raw content for bulk analysis
                source_content = f"=== SOURCE {i} ({action_name}){credibility_info} ===\n"
                source_content += f"{action_name} Content from {getattr(raw_result, 'url', 'unknown URL')}:\n"
                source_content += full_content
                all_raw_content.append(source_content)
                source_actions.append(action_name)
        
        if not all_raw_content:
            return self._create_empty_evidence("No content available from social media sources")
        
        # Combine all raw content for bulk analysis
        combined_sources = "\n\n".join(all_raw_content)
        
        # Log aggregation details
        self._log_aggregation_input(all_raw_content, source_actions, combined_sources)
        
        # Use LLM to analyze all social media content together
        try:
            takeaways = self._generate_bulk_analysis(combined_sources, doc)
            
            # Handle various forms of empty or useless responses
            if not takeaways or takeaways.strip() == "" or takeaways.strip().upper() == "NONE":
                self.logger.warning("🔍 LLM returned empty or 'NONE' response - providing fallback analysis")
                takeaways = f"Analyzed {len(all_raw_content)} social media sources ({', '.join(set(source_actions))}), but no specific insights were extracted by the AI model."
                
        except Exception as e:
            self.logger.warning(f"Failed to generate bulk social media analysis: {e}")
            takeaways = f"Analysis of {len(all_raw_content)} social media sources ({', '.join(set(source_actions))}) completed, but detailed extraction failed due to error: {str(e)[:100]}"
        
        return self._create_final_evidence(combined_sources, takeaways)

    def _extract_content_from_result(self, raw_result, action_name: str) -> str:
        """Extract full content from a social media result object."""
        if hasattr(raw_result, 'content') and raw_result.content:
            # This is a RedditResults/XResults object with a WebSource in content attribute
            web_source = raw_result.content
            if hasattr(web_source, 'content') and web_source.content:
                # Get the actual MultimodalSequence content from the WebSource
                content = web_source.content
                
                # Check if this is a MultimodalSequence with video objects
                if hasattr(content, 'data'):
                    # Count media items for logging
                    from ezmm import Video, Image
                    video_count = sum(1 for item in content.data if isinstance(item, Video))
                    image_count = sum(1 for item in content.data if isinstance(item, Image))
                    
                    # Debug: Log all item types
                    item_types = [type(item).__name__ for item in content.data]
                    self.logger.info(f"📄 MultimodalSequence for {action_name}: {len(content.data)} items, types: {item_types}")
                    
                    if video_count > 0 or image_count > 0:
                        self.logger.info(f"🎬 FOUND MEDIA in {action_name}: {video_count} videos, {image_count} images")
                    
                    # Convert to string but preserve video/image references
                    full_content = str(content)
                    
                    # Debug: Check what's in the string content
                    if 'video:' in full_content.lower() or '<video' in full_content.lower():
                        self.logger.info(f"🎥 Video references found in {action_name} content: {full_content[:100]}...")
                    
                    # Log if we lost video information in string conversion
                    if video_count > 0 and '<video:' not in full_content and 'Video object' in full_content:
                        self.logger.warning(f"⚠️  Video content may have been lost in string conversion for {action_name}")
                    
                else:
                    full_content = str(content)
                    
                self.logger.info(f"📄 Extracted MultimodalSequence content for {action_name}: {len(full_content)} characters")
                return full_content
            else:
                # WebSource doesn't have content, use its string representation  
                full_content = str(web_source)
                self.logger.info(f"📄 Using WebSource string for {action_name}: {len(full_content)} characters")
                return full_content
        else:
            # Fallback to string representation of the raw result
            full_content = str(raw_result)
            self.logger.info(f"📄 Fallback content for {action_name}: {len(full_content)} characters")
            return full_content

    def _log_aggregation_input(self, all_raw_content: List[str], source_actions: List[str], combined_sources: str):
        """Log details about the aggregation input."""
        self.logger.info("📊 SOCIAL MEDIA AGGREGATION INPUT:")
        self.logger.info(f"Total sources: {len(all_raw_content)}")
        self.logger.info(f"Source actions: {source_actions}")
        self.logger.info(f"Combined content length: {len(combined_sources)} characters")
        self.logger.info("Combined content preview:")
        self.logger.info(combined_sources[:1000] + ("..." if len(combined_sources) > 1000 else ""))

    def _generate_bulk_analysis(self, combined_sources: str, doc: Report) -> str:
        """Generate bulk analysis using LLM."""
        from defame.prompts.prompts import AnalyzeBulkSocialMediaPrompt
        from defame.common.modeling import make_model
        
        self.logger.info(f"🤖 Preparing bulk analysis for {len(combined_sources)} characters of social media content")
        
        # Use a multimodal model for social media aggregation to handle videos/images
        # Check if current LLM supports multimodal content, otherwise use GPT-4o
        llm_to_use = self.llm
        if not self.llm or not self._model_supports_multimodal(self.llm):
            self.logger.info("🎥 Switching to multimodal model (GPT-4o) for social media aggregation with video/image content")
            try:
                llm_to_use = make_model("gpt_4o")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not create multimodal model, using original LLM: {e}")
                llm_to_use = self.llm
        
        if not llm_to_use:
            raise Exception("No LLM available for social media aggregation")
            
        bulk_prompt = AnalyzeBulkSocialMediaPrompt(combined_sources, doc)
        
        # Validate the prompt is properly constructed
        prompt_str = str(bulk_prompt)
        if not prompt_str or len(prompt_str.strip()) < 50:
            self.logger.error(f"❌ Generated prompt is too short or empty: {len(prompt_str)} characters")
            return "Error: Unable to generate proper prompt for social media analysis"
        
        # Log the full prompt being sent to LLM
        self.logger.info("🔍 BULK SOCIAL MEDIA ANALYSIS PROMPT:")
        self.logger.info("=" * 80)
        self.logger.info(prompt_str)
        self.logger.info("=" * 80)
        
        self.logger.info("🚀 Sending bulk analysis request to LLM...")
        try:
            takeaways = llm_to_use.generate(bulk_prompt)
        except Exception as e:
            self.logger.error(f"❌ LLM generation failed: {e}")
            # Fallback: Try without multimedia content if the model still fails
            if "<video:" in combined_sources or "<image:" in combined_sources:
                self.logger.info("🔄 Attempting fallback with multimedia content descriptions...")
                sanitized_sources = self._convert_multimedia_to_descriptions(combined_sources)
                try:
                    bulk_prompt_fallback = AnalyzeBulkSocialMediaPrompt(sanitized_sources, doc)
                    takeaways = llm_to_use.generate(bulk_prompt_fallback)
                    self.logger.info("✅ Fallback analysis succeeded with multimedia descriptions")
                except Exception as e2:
                    self.logger.error(f"❌ Even fallback analysis failed: {e2}")
                    return "Error: Unable to generate bulk social media analysis - both multimodal and text-only approaches failed."
            else:
                return f"Error: Unable to generate bulk social media analysis: {e}"
        
        # Log the LLM response with better empty response handling
        response_length = len(str(takeaways).strip()) if takeaways else 0
        
        if response_length == 0:
            self.logger.warning("⚠️ LLM RETURNED EMPTY RESPONSE FOR BULK ANALYSIS:")
            self.logger.warning("-" * 80)
            self.logger.warning(f"Response length: {response_length} characters")
            self.logger.warning(f"Response content: '{takeaways}'")
            self.logger.warning("This may indicate an issue with the prompt or LLM configuration")
            self.logger.warning("-" * 80)
        else:
            self.logger.info("✅ LLM RESPONSE FOR BULK ANALYSIS:")
            self.logger.info("-" * 80)
            self.logger.info(f"Response length: {response_length} characters")
            self.logger.info(f"Response content: {takeaways}")
            self.logger.info("-" * 80)
        
        return takeaways
    
    def _model_supports_multimodal(self, model) -> bool:
        """Check if the model supports multimodal content."""
        if hasattr(model, 'model_name'):
            model_name = model.model_name.lower()
            # GPT-4o models support multimodal, GPT-4o-mini does not support videos
            return 'gpt-4o-2024' in model_name and 'mini' not in model_name
        return False
    
    def _convert_multimedia_to_descriptions(self, content: str) -> str:
        """Convert multimedia tags to text descriptions as fallback."""
        import re
        
        # Replace video tags with descriptive text
        content = re.sub(r'<video:\d+>', '[VIDEO: Social media video content - unable to process in text-only mode]', content)
        
        # Replace image tags with descriptive text
        content = re.sub(r'<image:\d+>', '[IMAGE: Social media image content - unable to process in text-only mode]', content)
        
        return content

    def _create_empty_evidence(self, message: str) -> Evidence:
        """Create empty evidence when no social media content is available."""
        class EmptySocialMediaAction:
            name = "AggregatedSocialMedia"
        
        class SimpleResults:
            def __init__(self, content):
                self.content = content
            def __str__(self):
                return self.content
        
        empty_action = EmptySocialMediaAction()
        empty_results = SimpleResults(message)
        
        return Evidence(
            raw=empty_results,
            action=empty_action,
            takeaways=MultimodalSequence([f"Social media aggregation: {message}"])
        )

    def _create_final_evidence(self, combined_sources: str, takeaways: str) -> Evidence:
        """Create the final aggregated evidence object."""
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
        
        # CRITICAL: Convert takeaways string to plain text to prevent video/image re-processing
        # The LLM response contains video references like "<video:75>" but these should NOT
        # trigger frame extraction again - they're just text placeholders in the analysis.
        # We need to escape or replace these references to prevent MultimodalSequence from parsing them.
        import re
        # Replace video/image references with text-only placeholders to prevent re-processing
        safe_takeaways = re.sub(r'<(video|image):(\d+)>', r'[\1:\2]', str(takeaways))
        
        return Evidence(
            raw=aggregated_results,
            action=aggregated_action,
            takeaways=MultimodalSequence([safe_takeaways])
        )

    def perform(self, action: Action, summarize: bool = True, **kwargs) -> Evidence:
        """Perform aggregation of social media actions."""
        if isinstance(action, AggregateSocialMedia):
            return self._perform_aggregation(action, **kwargs)
        else:
            raise ValueError(f"SocialMediaAggregator cannot handle action type: {type(action)}")

    def reset(self):
        """Reset the aggregator (nothing to reset currently)."""
        pass

    def set_claim_id(self, claim_id: str):
        """Set claim ID (not used by aggregator currently)."""
        pass

    def get_stats(self):
        """Get tool statistics."""
        return {"aggregations_performed": 0}  # Could track this in the future