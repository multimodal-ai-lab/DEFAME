from typing import Optional
from dataclasses import dataclass
import traceback

from .tool import Tool, Action
from ..integrations.social_media.x import x as x_integration
from ..integrations.search.common import WebSource
from ...common import Claim, Results, Evidence, Report, logger
from ...prompts.prompts import SummarizeSocialMediaSourcePrompt
from ezmm import MultimodalSequence
from openai import APIError
from jinja2.exceptions import TemplateSyntaxError

class SearchX(Action):
    """Retrieves and analyzes content from X (Twitter) posts and profiles."""
    name = "search_x"
    
    def __init__(self, url: str):
        """
        @param url: The full URL of the X post or profile to retrieve.
        """
        self._save_parameters(locals())
        self.url = url

    def __eq__(self, other):
        return isinstance(other, SearchX) and self.url == other.url

    def __hash__(self):
        return hash((self.name, self.url))


@dataclass
class XResults(Results):
    """Results from X content retrieval."""
    content: Optional[WebSource]
    url: str
    credibility_score: Optional[str] = None  # Changed to str for categorical scoring

    def __str__(self):
        if self.content is None:
            return f"No content found for {self.url}"
        
        result = f"X Content from {self.url}:\n"
        result += f"Content: {str(self.content)[:200]}...\n"
        if self.credibility_score is not None:
            result += f"Credibility Score: {self.credibility_score}\n"
        return result

    def is_useful(self) -> Optional[bool]:
        return self.content is not None

class XTool(Tool):
    """
    A tool for retrieving and analyzing content from X (formerly Twitter).
    It wraps the X integration and exposes it as a searchable action to the DEFAME framework.
    """
    name = "x_search"
    description = "A tool to retrieve content from X (formerly Twitter), including posts and user profiles."

    def __init__(self, llm=None, device=None, **kwargs):
        # XTool doesn't need llm or device parameters, but we accept them to be consistent with other tools
        super().__init__(llm, device)
        self.actions = [SearchX]
        
        # Register with SocialMediaAggregator
        from .social_media_aggregator import SocialMediaAggregator
        SocialMediaAggregator.register_social_media_tool(self, [SearchX])
    
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
            updated_action = SearchX(url=result.url)
            action = updated_action
        
        # Skip individual summarization for social media tools - bulk analysis will handle it
        summary = None
        return Evidence(result, action, takeaways=summary)
    
    def _perform(self, action: SearchX, claim_text: str) -> XResults:
        """Execute the SearchX action."""
        import asyncio
        import concurrent.futures
        import threading
        import sys
        import re
        
        # Use the actual claim text passed from the DEFAME fact-checker
        logger.info(f"X tool processing URL: {action.url} with claim: '{claim_text}'")
        
        # Check if this looks like a fictional URL and if we have real X URLs available
        original_url = action.url
        actual_url = self._find_real_x_url(action.url)
        if actual_url != original_url:
            logger.info(f"Replacing fictional URL {original_url} with real URL {actual_url}")
        
        try:
            def run_in_thread():
                """Run the async function in a separate thread with its own event loop."""
                # Check if we're on Windows and set the event loop policy
                if sys.platform.startswith('win'):
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                
                # Create new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # Ensure the integration has a fresh connection
                    return new_loop.run_until_complete(self._get_x_content_safely(actual_url, claim_text))
                finally:
                    # Properly close the loop and set it to None
                    try:
                        pending = asyncio.all_tasks(new_loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                        new_loop.run_until_complete(new_loop.shutdown_default_executor())
                    except Exception:
                        pass
                    finally:
                        new_loop.close()
                        asyncio.set_event_loop(None)
            
            # Always use threading approach to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                web_source = future.result(timeout=60)  # Increased timeout to 60 seconds
            
            # Extract credibility from WebSource if available
            credibility = getattr(web_source, 'credibility', None)
            
            return XResults(
                content=web_source,
                url=actual_url,  # Use the actual URL in results
                credibility_score=credibility
            )
        except Exception as e:
            print(f"Error retrieving X content: {e}")
            traceback.print_exc()
            # Return failed result
            return XResults(
                content=None,
                url=actual_url,  # Use the actual URL in results
                credibility_score=None
            )
    
    async def _get_x_content_safely(self, url: str, claim_text: str):
        """Safely get X content with proper session management."""
        # Create a fresh X instance for this thread to avoid session conflicts
        from ..integrations.social_media.x import X
        fresh_x = X()
        
        try:
            return await fresh_x.get(url, claim_text)
        finally:
            # Clean up the fresh instance
            await fresh_x.close()
    
    def _find_real_x_url(self, fictional_url: str) -> str:
        """Find real X/Twitter URLs from shared registry to replace fictional ones."""
        import re
        from ..shared_urls import get_x_urls
        
        # First check the shared URL registry (populated by search tools)
        x_urls = get_x_urls()
        print(f"🔍 Found {len(x_urls)} X URLs in shared registry: {x_urls}")
        
        if x_urls:
            # Create a simple round-robin mapping to avoid duplicates
            if not hasattr(self.__class__, '_url_mapping'):
                self.__class__._url_mapping = {}
                self.__class__._url_counter = 0
            
            # Check if we already have a mapping for this fictional URL
            if fictional_url in self.__class__._url_mapping:
                real_url = self.__class__._url_mapping[fictional_url]
                print(f"🔍 Using cached mapping: {fictional_url} -> {real_url}")
                return real_url
            
            # Map to next available URL using round-robin
            real_url = x_urls[self.__class__._url_counter % len(x_urls)]
            self.__class__._url_mapping[fictional_url] = real_url
            self.__class__._url_counter += 1
            
            print(f"🔍 Replacing fictional X URL {fictional_url} with real URL {real_url}")
            return real_url
        
        # Fallback: check document evidence (if any exists)
        if hasattr(self, 'current_doc') and self.current_doc:
            evidence_list = self.current_doc.get_all_evidence()
            print(f"🔍 Checking {len(evidence_list)} evidence items for X URLs as fallback")
            
            real_x_urls = []
            for evidence in evidence_list:
                if hasattr(evidence, 'result') and hasattr(evidence.result, 'content'):
                    if hasattr(evidence.result.content, 'reference'):
                        ref = str(evidence.result.content.reference)
                        if ("twitter.com" in ref or "x.com" in ref) and re.match(r'https://(?:www\.)?(?:twitter|x)\.com/[^/]+/status/\d+', ref):
                            real_x_urls.append(ref)
                    # Check if content is a list of sources
                    elif hasattr(evidence.result.content, '__iter__'):
                        for item in evidence.result.content:
                            if hasattr(item, 'reference'):
                                ref = str(item.reference)
                                if ("twitter.com" in ref or "x.com" in ref) and re.match(r'https://(?:www\.)?(?:twitter|x)\.com/[^/]+/status/\d+', ref):
                                    real_x_urls.append(ref)
            
            if real_x_urls:
                print(f"🔍 Found fallback X URL: {real_x_urls[0]}")
                return real_x_urls[0]
        
        print(f"🔍 No real X URLs found, using original: {fictional_url}")
        return fictional_url
    
    def _summarize(self, result: XResults, **kwargs) -> Optional[str]:
        """Analyze X content using LLM to extract relevant takeaways for fact-checking."""
        logger.info(f"Starting X tool LLM-based summarization for URL: {result.url}")
        
        if result.content is None:
            logger.warning(f"No content available for summarization: {result.url}")
            return f"Failed to retrieve content from {result.url}"
        
        # Get document context for LLM analysis
        doc = kwargs.get('doc')
        if not doc:
            logger.warning("No document context provided for LLM summarization")
            return self._fallback_summarize(result)
        
        logger.debug(f"Using LLM to analyze X content for claim: {doc.claim}")
        

        # Use LLM to analyze the X content with specialized social media prompt
        try:
            # Create a specialized social media prompt - pass the MultimodalSequence, not WebSource
            prompt = SummarizeSocialMediaSourcePrompt(result.content.content, doc)
            
            # Generate LLM summary with error handling like in searcher
            summary = self.llm.generate(prompt, max_attempts=3)
            if not summary:
                logger.warning("LLM returned empty summary, using fallback")
                return self._fallback_summarize(result)
            
            # Add credibility assessment to the LLM-generated summary
            enhanced_summary = self._add_credibility_assessment(summary, result.credibility_score)
            
            logger.info(f"Completed LLM-based X summarization, summary length: {enhanced_summary}")
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
    
    def _fallback_summarize(self, result: XResults) -> str:
        """Fallback summarization without LLM when there are errors."""
        logger.debug("Using fallback summarization (no LLM analysis)")
        
        # Extract raw text from scrapeMM
        scraped_text = self._extract_scraped_text(result.content)
        
        # Add credibility assessment
        summary = self._add_credibility_assessment(scraped_text, result.credibility_score)
        
        return summary
    
    def _extract_scraped_text(self, web_source: WebSource) -> str:
        """Extract formatted text from scrapeMM MultimodalSequence."""
        scraped_text = ""
        if hasattr(web_source, 'content') and web_source.content:
            multimodal_seq = web_source.content
            logger.debug(f"MultimodalSequence type: {type(multimodal_seq)}")
            
            # Handle different MultimodalSequence formats
            if hasattr(multimodal_seq, '__iter__') and not isinstance(multimodal_seq, str):
                try:
                    scraped_text = str(multimodal_seq[0]) if len(multimodal_seq) > 0 else str(multimodal_seq)
                    logger.debug(f"Extracted text from list structure, length: {len(scraped_text)}")
                except (IndexError, TypeError) as e:
                    logger.warning(f"Error extracting text from list structure: {e}")
                    scraped_text = str(multimodal_seq)
            else:
                scraped_text = str(multimodal_seq)
                logger.debug(f"Extracted text directly, length: {len(scraped_text)}")
        else:
            logger.warning("No content found in WebSource")
        
        return scraped_text
    
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
        await x_integration.close()