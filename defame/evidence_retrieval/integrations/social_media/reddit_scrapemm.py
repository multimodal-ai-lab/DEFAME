"""
Reddit integration using scrapeMM for content retrieval.
This is a cleaner, simplified version of the Reddit integration.
"""
import asyncio
import ssl
from typing import Optional
import aiohttp

from ezmm import MultimodalSequence
from defame.evidence_retrieval.integrations.integration import RetrievalIntegration
from defame.common.logger import logger

# Lazy import to avoid initialization issues
ScrapeMM_Reddit = None
SCRAPEMM_AVAILABLE = False


class RedditScrapeMM(RetrievalIntegration):
    """
    Reddit integration powered by scrapeMM.
    Retrieves posts, comments, subreddits, and user profiles from Reddit.
    """
    name = "reddit_scrapemm"
    domains = ["reddit.com", "www.reddit.com"]

    def __init__(self):
        super().__init__()
        
        # Lazy import scrapeMM
        global ScrapeMM_Reddit, SCRAPEMM_AVAILABLE
        if ScrapeMM_Reddit is None:
            try:
                from scrapemm.integrations.reddit import Reddit as ScrapeMM_Reddit_Import
                ScrapeMM_Reddit = ScrapeMM_Reddit_Import
                SCRAPEMM_AVAILABLE = True
            except ImportError as e:
                SCRAPEMM_AVAILABLE = False
                raise ImportError(f"scrapeMM is required for Reddit integration. Install with: pip install scrapemm. Error: {e}")
        
        if not SCRAPEMM_AVAILABLE:
            raise ImportError("scrapeMM is required for Reddit integration. Install with: pip install scrapemm")
        
        self.scraper = ScrapeMM_Reddit()
        self.session: Optional[aiohttp.ClientSession] = None

    def _retrieve(self, url: str) -> Optional[MultimodalSequence]:
        """Synchronous wrapper for async Reddit content retrieval."""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop - create task in new thread with new event loop
                import concurrent.futures
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        # Create a fresh session for each thread to avoid "Event loop is closed" errors
                        result = new_loop.run_until_complete(self._async_retrieve_with_new_session(url))
                        return result
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=60)
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self._async_retrieve(url))
                
        except Exception as e:
            logger.error(f"Error retrieving Reddit content for {url}: {e}")
            return None

    async def _async_retrieve_with_new_session(self, url: str) -> Optional[MultimodalSequence]:
        """Async retrieval with a new session (for thread-safe execution)."""
        session = None
        try:
            # Create SSL context that doesn't verify certificates (for development)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            session = aiohttp.ClientSession(connector=connector)
            
            content = await self.scraper.get(url, session)
            
            # DEBUG: Log what we're returning
            if content:
                logger.info(f"🔍 REDDIT_SCRAPEMM: Retrieved content for {url}")
                metadata = getattr(content, 'metadata', {})
                logger.info(f"🔍 REDDIT_SCRAPEMM: Metadata has {len(metadata)} keys")
                logger.info(f"🔍 REDDIT_SCRAPEMM: Metadata keys: {list(metadata.keys())}")
                logger.info(f"🔍 REDDIT_SCRAPEMM: 'comments' in metadata: {'comments' in metadata}")
                if 'comments' in metadata:
                    comments = metadata.get('comments', [])
                    logger.info(f"🔍 REDDIT_SCRAPEMM: Number of comments: {len(comments)}")
                    if comments:
                        logger.info(f"🔍 REDDIT_SCRAPEMM: First comment sample: {str(comments[0])[:200]}")
            
            return content
        except Exception as e:
            logger.error(f"Error in async Reddit retrieval for {url}: {e}")
            return None
        finally:
            if session and not session.closed:
                await session.close()

    async def _async_retrieve(self, url: str) -> Optional[MultimodalSequence]:
        """Async retrieval of Reddit content using scrapeMM."""
        try:
            session = await self._get_session()
            content = await self.scraper.get(url, session)
            return content
        except Exception as e:
            logger.error(f"Error in async Reddit retrieval for {url}: {e}")
            return None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with SSL context for Reddit."""
        if self.session is None or self.session.closed:
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Cleanup on deletion."""
        if self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass


# Singleton instance creation is delayed - will be created on first use  
reddit_scrapemm = None

def get_reddit_scrapemm():
    """Get or create the Reddit scrapeMM singleton."""
    global reddit_scrapemm
    if reddit_scrapemm is None:
        try:
            reddit_scrapemm = RedditScrapeMM()
        except ImportError:
            reddit_scrapemm = None
    return reddit_scrapemm
