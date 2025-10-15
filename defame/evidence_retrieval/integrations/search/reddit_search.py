"""
Reddit search platform using scrapeMM.
Searches Reddit for relevant posts and returns URLs for the RedditTool to analyze.
"""
import asyncio
from typing import Optional
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from defame.common import logger
from defame.evidence_retrieval.integrations.search.common import SearchResults, Query, WebSource
from defame.evidence_retrieval.integrations.search.remote_search_platform import RemoteSearchPlatform

# Lazy import to avoid initialization issues
ScrapeMM_Reddit = None
SCRAPEMM_AVAILABLE = False


class RedditSearch(RemoteSearchPlatform):
    """
    Search platform for Reddit that uses scrapeMM's search functionality.
    Returns URLs of relevant posts that can be analyzed by RedditTool.
    """
    name = "reddit"
    description = "Search Reddit for posts, discussions, and communities"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Lazy import scrapeMM
        global ScrapeMM_Reddit, SCRAPEMM_AVAILABLE
        if ScrapeMM_Reddit is None:
            try:
                from scrapemm.integrations.reddit import Reddit as ScrapeMM_Reddit_Import
                ScrapeMM_Reddit = ScrapeMM_Reddit_Import
                SCRAPEMM_AVAILABLE = True
            except ImportError as e:
                SCRAPEMM_AVAILABLE = False
                logger.warning(f"scrapeMM not available for Reddit search: {e}")
        
        if not SCRAPEMM_AVAILABLE:
            logger.warning("Reddit search platform disabled: scrapeMM not installed")
            self.scraper = None
        else:
            self.scraper = ScrapeMM_Reddit()
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _call_api(self, query: Query) -> Optional[SearchResults]:
        """Search Reddit using scrapeMM and return URLs as WebSource objects."""
        if not SCRAPEMM_AVAILABLE or self.scraper is None:
            logger.error("Cannot search Reddit: scrapeMM not available")
            return None
        
        if not query.has_text():
            logger.warning("Reddit search requires a text query")
            return None
        
        try:
            # Run async search in a way that's safe for nested event loops
            urls = self._run_async_search(query)
            
            if not urls:
                logger.info(f"No results found on Reddit for query: {query.text}")
                return SearchResults(sources=[], query=query)
            
            # Convert URLs to WebSource objects
            sources = [WebSource(reference=url) for url in urls]
            logger.info(f"Found {len(sources)} Reddit posts for query: {query.text}")
            
            return SearchResults(sources=sources, query=query)
            
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            return None

    def _run_async_search(self, query: Query) -> list[str]:
        """Run the async search in a way that handles nested event loops."""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, run in a separate thread
                future = self._executor.submit(self._search_in_new_loop, query)
                return future.result(timeout=60)
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self._async_search(query))
        except Exception as e:
            logger.error(f"Error in async Reddit search: {e}")
            return []

    def _search_in_new_loop(self, query: Query) -> list[str]:
        """Create a new event loop and run the search."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(self._async_search(query))
        finally:
            new_loop.close()

    async def _async_search(self, query: Query) -> list[str]:
        """Perform the actual async search using scrapeMM."""
        try:
            session = await self._get_session()
            limit = query.limit or self.max_search_results
            
            # Use scrapeMM's search method
            urls = await self.scraper.search(
                query=query.text,
                session=session,
                max_results=limit
            )
            
            return urls if urls else []
            
        except Exception as e:
            logger.error(f"Error in Reddit search API call: {e}")
            return []

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with SSL context for Reddit."""
        if self.session is None or self.session.closed:
            # Use the scraper's SSL context if available
            if hasattr(self.scraper, '_create_ssl_context'):
                connector = aiohttp.TCPConnector(ssl=self.scraper._create_ssl_context())
                self.session = aiohttp.ClientSession(connector=connector)
            else:
                self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Cleanup on deletion."""
        # Clean up executor
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        
        # Clean up session
        if self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass
