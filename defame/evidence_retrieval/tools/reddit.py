from typing import Optional
from dataclasses import dataclass
from .tool import Tool, Action
from ..integrations.social_media.reddit import reddit as reddit_integration
from ..integrations.search.common import WebSource
from ...common import Claim, Results, Evidence, Report, logger
from ...prompts.prompts import SummarizeSocialMediaSourcePrompt
from ezmm import MultimodalSequence
from openai import APIError
from jinja2.exceptions import TemplateSyntaxError

class SearchReddit(Action):
    """Retrieves and analyzes content from Reddit posts, comments, and profiles."""
    name = "search_reddit"
    
    def __init__(self, url: str):
        """
        @param url: The full URL of the Reddit post, comment, or profile to retrieve.
        """
        self._save_parameters(locals())
        self.url = url

    def __eq__(self, other):
        return isinstance(other, SearchReddit) and self.url == other.url

    def __hash__(self):
        return hash((self.name, self.url))


@dataclass
class RedditResults(Results):
    """Results from Reddit content retrieval."""
    content: Optional[WebSource]
    url: str
    credibility_score: Optional[str] = None  # Changed to str for categorical scoring

    def __str__(self):
        if self.content is None:
            return f"No content found for {self.url}"
        
        result = f"Reddit Content from {self.url}:\n"
        result += f"Content: {str(self.content)[:200]}...\n"
        if self.credibility_score is not None:
            result += f"Credibility Score: {self.credibility_score}\n"
        return result

    def is_useful(self) -> Optional[bool]:
        return self.content is not None

class RedditTool(Tool):
    """
    A tool for retrieving and analyzing content from Reddit.
    It wraps the Reddit integration and exposes it as a searchable action to the DEFAME framework.
    """
    name = "reddit_search"
    description = "A tool to retrieve content from Reddit, including posts, comments, and user profiles."

    def __init__(self, llm=None, device=None, **kwargs):
        # RedditTool doesn't need llm or device parameters, but we accept them to be consistent with other tools
        super().__init__(llm, device)
        self.actions = [SearchReddit]
        
        # Register with SocialMediaAggregator
        from .social_media_aggregator import SocialMediaAggregator
        SocialMediaAggregator.register_social_media_tool(self, [SearchReddit])
    
    def perform(self, action: Action, summarize: bool = True, **kwargs) -> Evidence:
        """Override perform to extract claim from doc parameter."""
        doc = kwargs.get('doc')
        claim_text = str(doc.claim) if doc and doc.claim else "Unknown claim"
        
        # Store doc reference for URL checking
        self.current_doc = doc
        
        assert type(action) in self.actions, f"Forbidden action: {action}"
        result = self._perform(action, claim_text)
        
        # Update the action URL to reflect the actual URL used (in case fictional URL was replaced)
        if hasattr(result, 'url') and result.url != action.url:
            # Create a new action with the actual URL used
            updated_action = SearchReddit(url=result.url)
            action = updated_action
        
        # Skip individual summarization for social media tools - bulk analysis will handle it
        summary = None
        return Evidence(result, action, takeaways=summary)
    
    def _find_real_reddit_url(self, fictional_url: str) -> str:
        """Find real Reddit URLs from shared registry to replace fictional ones."""
        import re
        from ..shared_urls import get_reddit_urls
        
        # First check the shared URL registry (populated by search tools)
        reddit_urls = get_reddit_urls()
        print(f"🔍 Found {len(reddit_urls)} Reddit URLs in shared registry: {reddit_urls}")
        
        if reddit_urls:
            # Try to find a URL that hasn't been used yet
            if not hasattr(self, '_used_urls'):
                self._used_urls = set()
                
            # Find an unused URL, or cycle through if all have been used
            available_urls = [url for url in reddit_urls if url not in self._used_urls]
            if not available_urls:
                # All URLs have been used, reset and start over
                self._used_urls.clear()
                available_urls = reddit_urls
                
            real_url = available_urls[0]
            self._used_urls.add(real_url)
            print(f"🔍 Replacing fictional URL {fictional_url} with real URL {real_url}")
            return real_url
        
        # Fallback: check document evidence (if any exists)
        if hasattr(self, 'current_doc') and self.current_doc:
            evidence_list = self.current_doc.get_all_evidence()
            print(f"🔍 Checking {len(evidence_list)} evidence items for Reddit URLs as fallback")
            
            real_reddit_urls = []
            for evidence in evidence_list:
                if hasattr(evidence, 'result') and hasattr(evidence.result, 'content'):
                    if hasattr(evidence.result.content, 'reference') and "reddit.com" in str(evidence.result.content.reference):
                        real_reddit_urls.append(str(evidence.result.content.reference))
                    # Check if content is a list of sources
                    elif hasattr(evidence.result.content, '__iter__'):
                        for item in evidence.result.content:
                            if hasattr(item, 'reference') and "reddit.com" in str(item.reference):
                                real_reddit_urls.append(str(item.reference))
            
            # Filter valid URLs
            valid_reddit_urls = [url for url in real_reddit_urls if re.match(r'https://(?:www\.)?reddit\.com/r/[^/]+/comments/[^/\s]+', url)]
            
            if valid_reddit_urls:
                print(f"🔍 Found fallback Reddit URL: {valid_reddit_urls[0]}")
                return valid_reddit_urls[0]
        
        print(f"🔍 No real Reddit URLs found, using original: {fictional_url}")
        return fictional_url
    
    def _perform(self, action: SearchReddit, claim_text: str) -> RedditResults:
        """Execute the SearchReddit action."""
        import asyncio
        import concurrent.futures
        import threading
        import re
        
        # Use the actual claim text passed from the DEFAME fact-checker
        logger.info(f"Reddit tool processing URL: {action.url} with claim: '{claim_text}'")
        
        # Check if this looks like a fictional URL and if we have real Reddit URLs available
        original_url = action.url
        actual_url = self._find_real_reddit_url(action.url)
        if actual_url != original_url:
            logger.info(f"Replacing fictional URL {original_url} with real URL {actual_url}")
        
        try:
            def run_in_thread():
                """Run the async function in a separate thread with its own event loop."""
                # Create new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # Create a fresh Reddit instance for this thread to avoid session conflicts
                    from ..integrations.social_media.reddit import Reddit
                    fresh_reddit = Reddit()
                    
                    result = new_loop.run_until_complete(fresh_reddit.get(actual_url, claim_text))
                    
                    # Clean up the fresh instance
                    new_loop.run_until_complete(fresh_reddit.close())
                    
                    return result
                finally:
                    new_loop.close()
            
            # Always use threading approach to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                web_source = future.result(timeout=30)  # 30 second timeout
            
            # Extract credibility from WebSource if available
            credibility = getattr(web_source, 'credibility', None)
            
            return RedditResults(
                content=web_source,
                url=actual_url,  # Use the actual URL in results
                credibility_score=credibility
            )
        except Exception as e:
            print(f"Error retrieving Reddit content: {e}")
            import traceback
            traceback.print_exc()
            # Return failed result
            return RedditResults(
                content=None,
                url=actual_url,  # Use the actual URL in results
                credibility_score=None
            )
    # use the text from scrapeMM
    def _summarize(self, result: RedditResults, **kwargs) -> Optional[str]:
        """Analyze Reddit content using LLM to extract relevant takeaways for fact-checking."""
        logger.info(f"Starting Reddit tool LLM-based summarization for URL: {result.url}")
        
        if result.content is None:
            logger.warning(f"No content available for summarization: {result.url}")
            return f"Failed to retrieve content from {result.url}"
        
        # Get document context for LLM analysis
        doc = kwargs.get('doc')
        if not doc:
            logger.warning("No document context provided for LLM summarization")
            return self._fallback_summarize(result)
        
        logger.debug(f"Using LLM to analyze Reddit content for claim: {doc.claim}")
        
        # Use LLM to analyze the Reddit content with specialized social media prompt
        try:
            # Create a specialized social media prompt
            prompt = SummarizeSocialMediaSourcePrompt(result.content.content, doc)
            
            # Generate LLM summary with error handling
            summary = self.llm.generate(prompt, max_attempts=3)
            if not summary:
                logger.warning("LLM returned empty summary, using fallback")
                return self._fallback_summarize(result)
            
            # Add credibility assessment to the LLM-generated summary
            enhanced_summary = self._add_credibility_assessment(summary, result.credibility_score)
            
            logger.info(f"Completed LLM-based Reddit summarization, summary length: {enhanced_summary}")
            return enhanced_summary
            
        except APIError as e:
            logger.info(f"APIError: {e} - Using fallback summarization for {result.url}")
            return self._fallback_summarize(result)
        except TemplateSyntaxError as e:
            logger.info(f"TemplateSyntaxError: {e} - Using fallback summarization for {result.url}")
            return self._fallback_summarize(result)
        except ValueError as e:
            logger.warning(f"ValueError: {e} - Using fallback summarization for {result.url}")
            return self._fallback_summarize(result)
        except Exception as e:
            logger.error(f"Error during LLM summarization: {e} - Using fallback for {result.url}")
            return self._fallback_summarize(result)
    
    def _fallback_summarize(self, result: RedditResults) -> str:
        """Fallback summarization without LLM when there are errors."""
        logger.debug("Using fallback summarization (no LLM analysis)")
        
        # Extract raw text from Reddit content
        content_text = self._extract_reddit_text(result.content)
        
        # Add credibility assessment
        summary = self._add_credibility_assessment(content_text, result.credibility_score)
        
        return summary
    
    def _extract_reddit_text(self, web_source: WebSource) -> str:
        """Extract formatted text from Reddit MultimodalSequence."""
        content_text = ""
        if hasattr(web_source, 'content') and web_source.content:
            multimodal_seq = web_source.content
            logger.debug(f"MultimodalSequence type: {type(multimodal_seq)}")
            
            # Reddit MultimodalSequence can contain either:
            # 1. A list where first element is the formatted text (posts with media)
            # 2. Just the formatted text directly (text-only posts or profiles)
            if hasattr(multimodal_seq, '__iter__') and not isinstance(multimodal_seq, str):
                try:
                    content_text = str(multimodal_seq[0]) if len(multimodal_seq) > 0 else str(multimodal_seq)
                    logger.debug(f"Extracted text from list structure, length: {len(content_text)}")
                except (IndexError, TypeError) as e:
                    logger.warning(f"Error extracting text from list structure: {e}")
                    content_text = str(multimodal_seq)
            else:
                content_text = str(multimodal_seq)
                logger.debug(f"Extracted text directly, length: {len(content_text)}")
        else:
            logger.warning("No content found in WebSource")
        
        return content_text
    
    def _add_credibility_assessment(self, content: str, credibility_score: Optional[str]) -> str:
        """Add credibility assessment with description to content."""
        if credibility_score is None:
            logger.warning("No credibility score available")
            return content
        
        # Credibility descriptions for fact-checking guidance
        credibility_descriptions = {
            "entirely credible": "High credibility sources - Give these sources primary weight in your decision",
            "mostly credible": "High credibility sources - Give these sources primary weight in your decision", 
            "moderately credible": "Moderate credibility sources - Consider these as supporting evidence",
            "somewhat credible": "Low credibility sources - Use with caution and only as secondary evidence",
            "entirely uncredible": "Low credibility sources - Use with caution and only as secondary evidence"
        }
        
        credibility_desc = credibility_descriptions.get(credibility_score, "Unknown credibility level")
        enhanced_content = f"{content}\n\n**Credibility Assessment:** {credibility_score.title()}\n{credibility_desc}"
        
        logger.info(f"Added credibility assessment: {credibility_score}")
        return enhanced_content

    async def close(self):
        """Ensures the underlying session is closed when the tool is shut down."""
        await reddit_integration.close()
