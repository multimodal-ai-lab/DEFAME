from typing import Optional, Any

from defame.common import MultimediaSnippet, Model
from defame.tools.tool import Tool
from defame.tools.social_media.platform_registry import get_handler_for_platform, get_supported_platforms
from defame.common.results import Result
from defame.common import logger

# Import the action and result classes we defined earlier
from defame.tools.social_media.common import RetrieveSocialMediaPost, SocialMediaPostResult, get_platform


class SocialMediaScraper(Tool):
    """Base class for social media platform scrapers. Provides interfaces for retrieving posts
    and profiles from various social media platforms."""
    
    name = "social_media_scraper"
    actions = [RetrieveSocialMediaPost]
    summarize = True  # Enable summarization for social media content
    
    # Dictionary mapping platform names to their API implementations
    _platform_handlers: dict[str, 'SocialMediaScraper'] = {}
    
    def __init__(self, llm: Model = None, api_keys: dict[str, str] = None, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.api_keys = api_keys or {}

    def supported_platforms(self) -> list[str]:
        """Returns a list of platforms supported by this scraper implementation."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _perform(self, action: RetrieveSocialMediaPost) -> Result:
        """Route the action to the appropriate platform-specific implementation."""
        handler = get_handler_for_platform(action.platform)
        if handler and handler != self:
            return handler._perform(action)

        logger.warning(f"No specialized handler found for platform '{action.platform}' for url {action.url}.")
        return self._create_error_result(action.url, "Platform not supported.")
    
    def _create_error_result(self, url: str, error_message: str) -> SocialMediaPostResult:
        """Create a SocialMediaPostResult that indicates an error occurred."""
        return SocialMediaPostResult(
            platform="unknown",
            post_url=url,
            author_username="",
            post_text=f"Error: {error_message}",
        )
    
    def retrieve_post(self, url: str) -> SocialMediaPostResult:
        """Retrieve a post from the given URL and return structured data."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _summarize(self, result: Result, **kwargs) -> Optional[MultimediaSnippet]:
        """Create a summary of the retrieved social media content."""
        if isinstance(result, SocialMediaPostResult):
            # For posts, we can directly use the text representation which already includes
            # key information like author, post content, engagement metrics, and media references
            return MultimediaSnippet(result.text)
        return None
    
    def get_stats(self) -> dict[str, Any]:
        """Returns the tool's usage statistics as a dictionary."""
        return {
            "API calls": getattr(self, "api_calls", 0),
            "Cache hits": getattr(self, "cache_hits", 0),
            "Errors": getattr(self, "errors", 0)
        }


def is_supported_sm_platform(url: str):
    """Check if the URL is for a supported platform."""
    return get_platform(url) in get_supported_platforms()