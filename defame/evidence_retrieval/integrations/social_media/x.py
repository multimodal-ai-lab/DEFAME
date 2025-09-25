import asyncio
import ssl
from typing import Optional
import aiohttp

from defame.evidence_retrieval.integrations.integration import RetrievalIntegration
from defame.evidence_retrieval.integrations.search.common import WebSource
from scrapemm.integrations.x import X as ScrapeMM_X
from defame.config import api_keys
from defame.common.log import logger
from defame.analysis.stance import init_stance_detector, StanceLabel, detect_comments_stance_batch
from defame.common.modeling import make_model
from ezmm import MultimodalSequence


def _calculate_reply_weighted_stance(comments_data: list, original_claim: str, post_content: str) -> float:
    """
    Calculates a weighted stance score based on credible comments using batch processing.
    Returns a score from -1.0 (all credible comments refute) to +1.0 (all credible comments support)
    """
    if not comments_data:
        return 0.0
    
    # Use batch stance detection for all comments at once
    try:
        stance_results = detect_comments_stance_batch(comments_data[:20], original_claim, post_content)
        
        if not stance_results.get("results"):
            logger.warning("No stance results returned from batch processing")
            return 0.0
        
        total_weight = 0
        weighted_stance_sum = 0
        
        # Credibility weights for comment authors
        credibility_weights = {
            "entirely credible": 1.0,
            "mostly credible": 0.8,
            "moderately credible": 0.6,
            "somewhat credible": 0.4,
            "entirely uncredible": 0.1
        }
        
        # Process each result from batch stance detection
        for result in stance_results["results"]:
            try:
                comment_id = result.get("comment_id")  # Uses same field name for consistency
                stance_label = result.get("stance")
                reason = result.get("reason", "")
                confidence = result.get("confidence", "MEDIUM")
                
                # Find the corresponding comment data to get author info
                comment_data = None
                for comment in comments_data[:20]:
                    if comment.get('id', f"comment_{comments_data.index(comment)}") == comment_id:
                        comment_data = comment
                        break
                
                if not comment_data:
                    continue
                
                # Calculate comment author credibility (simplified for X since we don't have detailed author metadata)
                comment_author_metadata = comment_data.get('author_metadata', {})
                if comment_author_metadata:
                    comment_credibility = calculate_profile_credibility(comment_author_metadata)
                    weight = credibility_weights.get(comment_credibility, 0.5)
                else:
                    # Default weight based on likes if no author metadata
                    likes = comment_data.get('likes', 0)
                    if likes > 100:
                        weight = 0.8
                    elif likes > 10:
                        weight = 0.6
                    else:
                        weight = 0.4
                
                # Adjust weight based on confidence
                if confidence == "HIGH":
                    weight *= 1.0
                elif confidence == "MEDIUM":
                    weight *= 0.8
                elif confidence == "LOW":
                    weight *= 0.6
                
                # Convert stance to numerical value
                if stance_label == StanceLabel.SUPPORTS:
                    stance_value = 1.0
                elif stance_label == StanceLabel.REFUTES:
                    stance_value = -1.0
                else:  # NEUTRAL or UNRELATED
                    stance_value = 0.0
                    
                # Add weighted contribution
                weighted_stance_sum += weight * stance_value
                total_weight += weight
                
                logger.debug(f"Comment {comment_id}: stance={stance_label.value}, likes={comment_data.get('likes', 0)}, weight={weight:.2f}, reason='{reason}'")
                
            except Exception as e:
                logger.debug(f"Error processing individual stance result: {e}")
                continue
        
        # Return normalized weighted stance
        if total_weight > 0:
            final_score = weighted_stance_sum / total_weight
            logger.info(f"Calculated weighted stance: {final_score:.3f} from {len(stance_results['results'])} comments")
            return final_score
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Error in batch reply stance calculation: {e}")
        return 0.0

def _calculate_base_author_score(metadata: dict) -> int:
    """
    Calculates a score component based on author-level metrics (no engagement).
    
    CREDIBILITY WEIGHTING SYSTEM:
    - Author credibility: 75 points (75% weight) - based on verification, followers, account integrity
    - Comment stance: 25 points (25% weight) - based on weighted analysis of reply credibility
    - Total max score: 100 points for balanced evaluation
    """
    score = 0
    author_metrics = metadata.get("author_public_metrics", {})
    
    # Verification Status (primary credibility indicator)
    if metadata.get("author_verified"):
        if metadata.get("author_verified_type") == "gold":
            score += 50  # Official organization - highest credibility
        elif metadata.get("author_verified_type") == "government":
            score += 50  # Government account - highest credibility
        elif metadata.get("author_verified_type") == "blue":
            score += 25  # Paid verification - moderate credibility
        else:
            score += 15  # Generic verified - some credibility
            
    # Account establishment (NOT popularity) - indicates sustained presence
    followers = author_metrics.get("followers_count", 0)
    if followers > 10000:
        score += 10  # Established presence
    elif followers > 1000:
        score += 5   # Some establishment
    
    # Account age/creation (if available in metadata)
    if metadata.get("author_created_at"):
        # Could add logic to calculate account age for establishment
        pass

    # Account integrity - reward clean accounts, penalize problematic ones
    has_negative_flags = False
    
    if metadata.get("author_protected"):
        score -= 25  # Private account - less transparent
        has_negative_flags = True
    if metadata.get("author_withheld"):
        score -= 60  # Content withheld - serious credibility issues
        has_negative_flags = True
    if metadata.get("author_suspended"):
        score -= 70  # Suspended account - major credibility issues
        has_negative_flags = True
    
    # Reward accounts with clean integrity (no negative flags)
    if not has_negative_flags:
        score += 15  # Clean account bonus - completes the 75-point scale
        
    return score

def _score_to_category(score: int, max_score: int) -> str:
    """Maps a numerical score to a credibility category."""
    normalized_score = max(0, min(1, score / max_score))
    
    if normalized_score >= 0.8:
        return "entirely credible"
    elif normalized_score >= 0.6:
        return "mostly credible"
    elif normalized_score >= 0.4:
        return "moderately credible"
    elif normalized_score >= 0.2:
        return "somewhat credible"
    else:
        return "entirely uncredible"

def calculate_profile_credibility(metadata: dict) -> str:
    """
    Calculates a credibility category for a user profile.
    Returns one of: "entirely uncredible", "somewhat credible", 
    "moderately credible", "mostly credible", "entirely credible"
    """
    score = _calculate_base_author_score(metadata)
    max_score = 75  # Max score: 50 (verification) + 10 (followers) + 15 (clean account)
    
    return _score_to_category(score, max_score)

def calculate_post_credibility(metadata: dict, comments_data: list = None, claim: str = "", post_content: str = "") -> str:
    """
    Calculates a credibility category for a specific post with optional comment weighting.
    Returns one of: "entirely uncredible", "somewhat credible", 
    "moderately credible", "mostly credible", "entirely credible"
    """
    # Start with base author score (max 75 points = 75% of total weight)
    score = _calculate_base_author_score(metadata)
    
    # Add weighted comment stance if available (max 25 points = 25% of total weight)
    if comments_data and claim:
        try:
            comment_stance_score = _calculate_reply_weighted_stance(comments_data, claim, post_content)
            
            # Convert stance to credibility modifier (scaled to match our scoring system)
            # Positive stance (credible comments support) = higher credibility
            # Negative stance (credible comments refute) = lower credibility  
            stance_modifier = comment_stance_score * 25  # 25-point influence (25% weight) on 100-point scale
            score += stance_modifier
            
            logger.debug(f"Comment stance modifier: {stance_modifier:.1f} points (from {len(comments_data)} comments)")
            
        except Exception as e:
            logger.warning(f"Error calculating comment stance weighting: {e}")
    
    max_score = 100  # 75 (author) + 25 (stance) = 100 total
    return _score_to_category(score, max_score)

def calculate_post_credibility_simple(metadata: dict) -> str:
    """
    Simple post credibility calculation without reply analysis (fallback method).
    Uses only author-based credibility (75% weight) without comment stance (25% weight).
    """
    score = _calculate_base_author_score(metadata)
    max_score = 100  # Consistent with full credibility calculation
    return _score_to_category(score, max_score)


class X(RetrievalIntegration):
    """
    The X (Twitter) integration for DEFAME, powered by scrapeMM.
    This tool retrieves content from X and formats it into a WebSource object.
    """
    name = "x"
    is_free = False  # Requires API keys
    is_local = False
    domains = ["x.com", "twitter.com", "www.x.com", "www.twitter.com"]

    def __init__(self):
        super().__init__()
        self.scraper = ScrapeMM_X()
        self.session: Optional[aiohttp.ClientSession] = None
        # Initialize stance detector with a default model
        self.llm = make_model("gpt_4o_mini")  # You can make this configurable
        init_stance_detector(self.llm)

    def _retrieve(self, url: str) -> Optional[MultimodalSequence]:
        """Synchronous wrapper for async X content retrieval."""
        try:
            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # We're in an event loop, create a new thread to run async code
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    # Create new event loop in thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self._async_retrieve(url))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=30)
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self._async_retrieve(url))
                
        except Exception as e:
            logger.error(f"Error retrieving X content for {url}: {e}")
            return None

    async def _async_retrieve(self, url: str) -> Optional[MultimodalSequence]:
        """Retrieves X content using scrapeMM integration."""
        try:
            session = await self._get_session()
            content = await self.scraper.get(url, session)
            return content
        except Exception as e:
            logger.error(f"Error in async X retrieval for {url}: {e}")
            return None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Creates an aiohttp session if one doesn't exist or is closed."""
        if self.session is None or self.session.closed:
            # Create SSL context that bypasses certificate verification for development
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Create connector with SSL bypass
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session

    async def _reset_session(self):
        """Force reset the session - useful for event loop issues."""
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None

    async def get(self, url: str, claim: str) -> Optional[WebSource]:
        """Asynchronously fetches content from an X URL and analyzes it."""
        logger.info(f"X integration processing URL: {url} with claim: '{claim}'")
        
        try:
            if not self.scraper.connected:
                await self.scraper.connect(
                    username=api_keys.get("twitter_username"),
                    password=api_keys.get("twitter_password"),
                    email=api_keys.get("twitter_email")
                )

            session = await self._get_session()
            content = await self.scraper.get(url, session)

            if not content:
                logger.warning(f"No content retrieved from {url}")
                return None

            # Extract metadata for analysis
            metadata = getattr(content, 'metadata', {})
            
            
            # Get comments data if available and convert from strings to structured objects
            raw_comments = metadata.get("comments", []) if metadata else []
            comments_data = []
            
            # Convert string comments to structured format expected by stance detection
            for i, comment_str in enumerate(raw_comments):
                if isinstance(comment_str, str):
                    # Parse comment string format: "Comment by @author (N likes):\ntext"
                    import re
                    match = re.match(r"Comment by @(\w+) \((\d+) likes\):\n(.+)", comment_str, re.DOTALL)
                    if match:
                        author, likes, text = match.groups()
                        comments_data.append({
                            'id': f"comment_{i}",
                            'text': text.strip(),
                            'author': author,
                            'likes': int(likes),
                            'author_metadata': {}  # No detailed author metadata available from X API currently
                        })
                    else:
                        # Fallback for unexpected format
                        comments_data.append({
                            'id': f"comment_{i}",
                            'text': comment_str,
                            'author': 'unknown',
                            'likes': 0,
                            'author_metadata': {}
                        })
            
            # Logic branches depending on whether it's a post or a profile
            if "post_public_metrics" in metadata:
                # This is a post - use enhanced credibility with comments if available
                if comments_data and claim:
                    # Get the actual tweet text for context
                    post_content = metadata.get("tweet_text", "")
                    credibility_category = calculate_post_credibility(metadata, comments_data, claim, post_content)
                else:
                    credibility_category = calculate_post_credibility_simple(metadata)
            else:
                # This is a profile
                credibility_category = calculate_profile_credibility(metadata)

            logger.info(f"Analyzed {url}: Credibility='{credibility_category}'")

            # Package into WebSource
            return WebSource(
                reference=url,
                content=content,
                title=f"X content from {url}",
                preview=str(content)[:200] + "...",
                credibility=credibility_category
            )
            
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.warning(f"Event loop closed error for {url}, resetting session...")
                await self._reset_session()
                return None
            else:
                raise e
        except Exception as e:
            logger.error(f"Error retrieving X content for {url}: {e}")
            return None

    def get_sync(self, url: str, claim: str) -> Optional[WebSource]:
        """
        Synchronously retrieves and analyzes content from an X URL.
        This is a convenience wrapper around the async get method.
        """
        return asyncio.run(self.get(url, claim))

    async def close(self):
        """Closes the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

# Create a singleton instance for easy import
x = X()
