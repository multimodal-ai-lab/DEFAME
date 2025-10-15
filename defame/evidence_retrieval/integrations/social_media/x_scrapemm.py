"""
X (Twitter) integration using scrapeMM for content retrieval.
This is a cleaner, simplified version of the X integration.
"""
import asyncio
import ssl
from typing import Optional
import aiohttp

from ezmm import MultimodalSequence
from defame.evidence_retrieval.integrations.integration import RetrievalIntegration
from defame.common.logger import logger

# Lazy import to avoid initialization issues
ScrapeMM_X = None
SCRAPEMM_AVAILABLE = False


class XScrapeMM(RetrievalIntegration):
    """
    X (Twitter) integration powered by scrapeMM.
    Retrieves posts and profiles from X/Twitter.
    """
    name = "x_scrapemm"
    domains = ["x.com", "twitter.com", "www.x.com", "www.twitter.com"]

    def __init__(self):
        super().__init__()
        
        # Lazy import scrapeMM
        global ScrapeMM_X, SCRAPEMM_AVAILABLE
        if ScrapeMM_X is None:
            try:
                from scrapemm.integrations.x import X as ScrapeMM_X_Import
                ScrapeMM_X = ScrapeMM_X_Import
                SCRAPEMM_AVAILABLE = True
            except ImportError as e:
                SCRAPEMM_AVAILABLE = False
                raise ImportError(f"scrapeMM is required for X integration. Install with: pip install scrapemm. Error: {e}")
        
        if not SCRAPEMM_AVAILABLE:
            raise ImportError("scrapeMM is required for X integration. Install with: pip install scrapemm")
        
        self.scraper = ScrapeMM_X()
        self.session: Optional[aiohttp.ClientSession] = None

    def _retrieve(self, url: str) -> Optional[MultimodalSequence]:
        """Synchronous wrapper for async X content retrieval."""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop - create task and wait for it
                # This is safe because the scraper is already managing the event loop
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
            logger.error(f"Error retrieving X content for {url}: {e}")
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
                logger.info(f"🔍 X_SCRAPEMM: Retrieved content for {url}")
                metadata = getattr(content, 'metadata', {})
                logger.info(f"🔍 X_SCRAPEMM: Metadata has {len(metadata)} keys")
                logger.info(f"🔍 X_SCRAPEMM: Metadata keys: {list(metadata.keys())}")
                logger.info(f"🔍 X_SCRAPEMM: 'comments' in metadata: {'comments' in metadata}")
                logger.info(f"🔍 X_SCRAPEMM: 'replies' in metadata: {'replies' in metadata}")
                if 'comments' in metadata:
                    comments = metadata.get('comments', [])
                    logger.info(f"🔍 X_SCRAPEMM: Number of comments: {len(comments)}")
                    if comments:
                        logger.info(f"🔍 X_SCRAPEMM: First comment sample: {str(comments[0])[:200]}")
                if 'replies' in metadata:
                    replies = metadata.get('replies', [])
                    logger.info(f"🔍 X_SCRAPEMM: Number of replies: {len(replies)}")
            
            return content
        except Exception as e:
            logger.error(f"Error in async X retrieval for {url}: {e}")
            return None
        finally:
            if session and not session.closed:
                await session.close()

    async def _async_retrieve(self, url: str) -> Optional[MultimodalSequence]:
        """Async retrieval of X content using scrapeMM."""
        try:
            session = await self._get_session()
            content = await self.scraper.get(url, session)
            return content
        except Exception as e:
            logger.error(f"Error in async X retrieval for {url}: {e}")
            return None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
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
x_scrapemm = None

def get_x_scrapemm():
    """Get or create the X scrapeMM singleton."""
    global x_scrapemm
    if x_scrapemm is None:
        try:
            x_scrapemm = XScrapeMM()
        except ImportError:
            x_scrapemm = None
    return x_scrapemm
