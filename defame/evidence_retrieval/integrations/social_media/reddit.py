import asyncio
import aiohttp
from typing import Optional

from defame.evidence_retrieval.integrations.integration import RetrievalIntegration
from defame.evidence_retrieval.integrations.search.common import WebSource
from scrapemm.integrations.reddit import Reddit as ScrapeMM_Reddit
from defame.common.log import logger
from defame.analysis.stance import init_stance_detector, StanceLabel, detect_comments_stance_batch
from defame.common.modeling import make_model
from ezmm import MultimodalSequence


class Reddit(RetrievalIntegration):
    """
    The Reddit integration for DEFAME, powered by scrapeMM.
    This tool retrieves content from Reddit and formats it into a WebSource object.
    """
    name = "reddit"
    is_free = False  # Requires API keys
    is_local = False
    domains = ["reddit.com", "www.reddit.com"]

    def __init__(self):
        super().__init__()
        self.scraper = ScrapeMM_Reddit()
        self.session: Optional[aiohttp.ClientSession] = None
        # Initialize stance detector with a default model
        self.llm = make_model("gpt_4o_mini")  # You can make this configurable
        init_stance_detector(self.llm)

    def _retrieve(self, url: str) -> Optional[MultimodalSequence]:
        """Synchronous wrapper for async Reddit content retrieval."""
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
            logger.error(f"Error retrieving Reddit content for {url}: {e}")
            return None

    async def _async_retrieve(self, url: str) -> Optional[MultimodalSequence]:
        """Retrieves Reddit content using scrapeMM integration."""
        try:
            session = await self._get_session()
            content = await self.scraper.get(url, session)
            return content
        except Exception as e:
            if "Event loop is closed" in str(e):
                logger.warning(f"Event loop closed error for {url}, creating fresh session...")
                try:
                    # Reset session and try again
                    await self._reset_session()
                    fresh_session = await self._get_session()
                    content = await self.scraper.get(url, fresh_session)
                    return content
                except Exception as retry_error:
                    logger.error(f"Retry failed for Reddit retrieval of {url}: {retry_error}")
                    return None
            else:
                logger.error(f"Error in async Reddit retrieval for {url}: {e}")
                return None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Creates an aiohttp session if one doesn't exist or is closed."""
        try:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()
            return self.session
        except Exception as e:
            # If there's an issue with session checking, create a fresh one
            logger.debug(f"Session access issue, creating fresh session: {e}")
            if self.session and not self.session.closed:
                try:
                    await self.session.close()
                except Exception:
                    pass
            self.session = aiohttp.ClientSession()
            return self.session

    async def _reset_session(self):
        """Force reset the session - useful for event loop issues."""
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None

    async def get(self, url: str, claim: str) -> Optional[WebSource]:
        """Asynchronously fetches content from a Reddit URL and analyzes it."""
        logger.info(f"Reddit integration processing URL: {url}")
        
        try:
            if not self.scraper.connected:
                logger.warning("❌ Reddit integration not connected in scrapeMM. Check secrets.")
                return None

            session = await self._get_session()
            content = await self.scraper.get(url, session)

            if not content:
                logger.warning(f"No content retrieved from {url}")
                return None

            # Extract metadata for analysis
            metadata = getattr(content, 'metadata', {})
            
            # Determine credibility based on content type using metadata
            if metadata.get("post_id"):
                # This is a Reddit post - use enhanced credibility with comments if available
                comments_data = metadata.get("comments", [])
                # Improved post content extraction with better fallbacks
                post_content = metadata.get("post_text", "")
                if not post_content and metadata.get("post_title"):
                    # If post_text is empty, use title + any selftext or description
                    post_content = metadata.get("post_title", "")
                    if metadata.get("selftext"):
                        post_content += f"\n\n{metadata.get('selftext')}"
                if not post_content:
                    # Final fallback: extract from the main content text
                    main_text = content.text if hasattr(content, 'text') else str(content)
                    # Extract the post content after "**title**" section if available
                    if "**" in main_text:
                        lines = main_text.split('\n')
                        post_lines = []
                        capture_post = False
                        for line in lines:
                            if line.startswith("**") and line.endswith("**") and len(line) > 4:
                                capture_post = True
                                post_lines.append(line.strip("*"))
                            elif capture_post and line.strip():
                                post_lines.append(line)
                            elif capture_post and not line.strip():
                                break
                        post_content = '\n'.join(post_lines) if post_lines else main_text[:500]
                    else:
                        post_content = main_text[:500]  # Use first 500 chars as fallback
                credibility_category = self._calculate_reddit_post_credibility_enhanced(metadata, comments_data, claim, post_content)
            elif metadata.get("username"):
                # This is a Reddit user profile
                credibility_category = self._calculate_reddit_user_credibility(metadata)
            elif metadata.get("subreddit_name"):
                # This is a Reddit subreddit
                credibility_category = self._calculate_reddit_subreddit_credibility(metadata)
            else:
                # Fallback to text-based detection for older data
                text_to_analyze = content.text if hasattr(content, 'text') else str(content)
                if text_to_analyze and any(keyword in text_to_analyze.lower() for keyword in ['reddit post', 'author:', 'posted:', 'upvotes']):
                    comments_data = metadata.get("comments", [])
                    post_content = metadata.get("post_text", text_to_analyze)
                    credibility_category = self._calculate_reddit_post_credibility_enhanced_fallback(comments_data, claim, post_content)
                elif text_to_analyze and 'reddit user' in text_to_analyze.lower():
                    credibility_category = self._calculate_reddit_user_credibility_fallback(text_to_analyze)
                elif text_to_analyze and 'reddit subreddit' in text_to_analyze.lower():
                    credibility_category = self._calculate_reddit_subreddit_credibility_fallback(text_to_analyze)
                else:
                    credibility_category = "moderately credible"

            logger.info(f"Analyzed {url}: Credibility='{credibility_category}'")

            # Package into WebSource
            return WebSource(
                reference=url,
                content=content,
                title=f"Reddit content from {url}",
                preview=str(content)[:200] + "...",
                credibility=credibility_category
            )
            
        except Exception as e:
            logger.error(f"Error processing Reddit content from {url}: {e}")
            return None

    def _score_to_category(self, score: float) -> str:
        """Maps a numerical score to a credibility category."""
        if score >= 0.8:
            return "entirely credible"
        elif score >= 0.6:
            return "mostly credible"
        elif score >= 0.4:
            return "moderately credible"
        elif score >= 0.2:
            return "somewhat credible"
        else:
            return "entirely uncredible"

    def _calculate_comment_weighted_stance(self, comments_data: list, original_claim: str, post_content: str) -> float:
        """
        Calculates a weighted stance score based on credible comments using batch processing.
        Returns a score from -1.0 (all credible comments refute) to +1.0 (all credible comments support)
        """
        if not comments_data:
            return 0.0
        
        # Use batch stance detection for all comments at once
        try:
            # Map Reddit comment structure to stance detection format
            mapped_comments = []
            for comment in comments_data[:20]:
                mapped_comment = {
                    'id': comment.get('id', f"comment_{len(mapped_comments)}"),
                    'text': comment.get('body', ''),  # Map 'body' to 'text'
                    'author': comment.get('author', 'unknown')
                }
                mapped_comments.append(mapped_comment)
            
            stance_results = detect_comments_stance_batch(mapped_comments, original_claim, post_content)

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
                    comment_id = result.get("comment_id")
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
                    
                    # Calculate comment author credibility using structured data if available
                    if 'author_info' in comment_data:
                        # Use structured author info from metadata
                        comment_author_text = comment_data.get('author_info', '')
                        comment_credibility = self._calculate_reddit_user_credibility_fallback(comment_author_text)
                    else:
                        # For structured comment data, we'd need actual user metadata
                        # For now, use basic heuristics based on score and awards
                        comment_score = comment_data.get('score', 0)
                        comment_awards = comment_data.get('awards', 0)
                        
                        # Simple heuristic based on comment engagement (fallback)
                        if comment_score > 100 and comment_awards > 0:
                            comment_credibility = "mostly credible"
                        elif comment_score > 50:
                            comment_credibility = "moderately credible"
                        elif comment_score > 10:
                            comment_credibility = "somewhat credible"
                        else:
                            comment_credibility = "moderately credible"  # Neutral default
                    weight = credibility_weights.get(comment_credibility, 0.5)
                    
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
                    
                    logger.debug(f"Comment {comment_id}: stance={stance_label.value}, credibility={comment_credibility}, weight={weight:.2f}, reason='{reason}'")
                    
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
            logger.error(f"Error in batch comment stance calculation: {e}")
            return 0.0

    def _calculate_base_post_score(self, metadata: dict) -> float:
        """Calculate base post credibility score using metadata."""
        score = 0.5  # Base score
        
        # Content quality indicators (NOT engagement)
        if metadata.get('is_original_content', False):
            score += 0.2  # Original content is more credible
        
        # Source attribution (credibility indicator)
        if metadata.get('url_linked') and not metadata.get('is_self', True):
            score += 0.15  # Has external sources
        
        # Moderator actions (content quality indicators)
        if metadata.get('locked', False):
            score -= 0.2  # Controversial/problematic content
        if metadata.get('archived', False):
            score += 0.05  # Established content that has been preserved
            
        # Content flags
        if metadata.get('over_18', False):
            score -= 0.1  # May be less reliable for fact-checking
            
        # Flair as credibility indicators
        flair_text = (metadata.get('link_flair_text') or '').lower()
        if any(keyword in flair_text for keyword in ['misleading', 'unverified', 'fake']):
            score -= 0.3  # Flagged as problematic
        if any(keyword in flair_text for keyword in ['verified', 'confirmed', 'official']):
            score += 0.2  # Verification indicators
        
        # Awards as quality indicators (NOT engagement)
        total_awards = metadata.get('total_awards_received', 0)
        if total_awards > 0:
            score += min(0.1, total_awards * 0.01)  # Cap at 0.1 bonus
        
        return score

    def _calculate_reddit_post_credibility_enhanced(self, metadata: dict, comments_data: list, claim: str, post_content: str) -> str:
        """
        Enhanced post credibility that includes weighted comment stance using metadata.
        
        CREDIBILITY WEIGHTING SYSTEM:
        - Post/author credibility: 80% weight (base score 0.0-1.0)
        - Comment stance: 20% weight (±0.2 modifier based on weighted reply analysis)
        - Total range: 0.0-1.0 for balanced evaluation
        """
        
        # Start with existing credibility factors (80% weight)
        score = self._calculate_base_post_score(metadata)
        
        # Add weighted comment stance (20% weight)
        if comments_data:
            try:
                comment_stance_score = self._calculate_comment_weighted_stance(comments_data, claim, post_content)

                # Convert stance to credibility modifier
                # Positive stance (credible comments support) = higher credibility
                # Negative stance (credible comments refute) = lower credibility
                stance_modifier = comment_stance_score * 0.2  # 20% influence
                score += stance_modifier
                
                logger.debug(f"Comment stance modifier: {stance_modifier:.3f} (from {len(comments_data)} comments)")
                
            except Exception as e:
                logger.warning(f"Error calculating comment stance weighting: {e}")
        
        score = max(0.0, min(1.0, score))
        return self._score_to_category(score)

    def _calculate_reddit_post_credibility_enhanced_fallback(self, comments_data: list, claim: str, post_content:str) -> str:
        """Fallback method using text parsing for older data."""
        
        # Start with existing credibility factors
        score = self._calculate_base_post_score_fallback(post_content)
        
        # Add weighted comment stance (if comments available)
        if comments_data:
            try:
                comment_stance_score = self._calculate_comment_weighted_stance(comments_data, claim, post_content)
                
                # Convert stance to credibility modifier
                stance_modifier = comment_stance_score * 0.2  # 20% influence
                score += stance_modifier
                
                logger.debug(f"Comment stance modifier: {stance_modifier:.3f} (from {len(comments_data)} comments)")
                
            except Exception as e:
                logger.warning(f"Error calculating comment stance weighting: {e}")
        
        score = max(0.0, min(1.0, score))
        return self._score_to_category(score)

    def _calculate_base_post_score_fallback(self, text: str) -> float:
        """Fallback method: Calculate base post credibility score using text parsing."""
        score = 0.5  # Base score
        
        import re
        
        # Content quality indicators (NOT engagement)
        if 'original content' in text.lower() or '[oc]' in text.lower():
            score += 0.2  # Original content is more credible
        
        # Source attribution (credibility indicator)
        if any(source in text.lower() for source in ['source:', 'via:', 'https://', 'http://']):
            score += 0.15  # Has external sources
        
        # Moderator actions (content quality indicators)
        if 'post locked by moderators' in text.lower():
            score -= 0.2  # Controversial/problematic content
        if 'removed by moderators' in text.lower():
            score -= 0.4  # Violated rules
        if 'moderator' in text.lower() and 'official' in text.lower():
            score += 0.25  # Official moderator post
            
        # Content flags
        if 'nsfw' in text.lower():
            score -= 0.1  # May be less reliable for fact-checking
        if 'misleading' in text.lower() or 'unverified' in text.lower():
            score -= 0.3  # Flagged as problematic
        if 'verified' in text.lower() or 'confirmed' in text.lower():
            score += 0.2  # Verification indicators
        
        # Awards as quality indicators (NOT engagement)
        award_keywords = ['gold', 'silver', 'platinum', 'award']
        if any(award in text.lower() for award in award_keywords):
            score += 0.1
        
        return score

    def _calculate_reddit_post_credibility(self, text: str, stance) -> str:
        """Calculate credibility category for Reddit posts based on text analysis (fallback method)."""
        score = self._calculate_base_post_score_fallback(text)
        score = max(0.0, min(1.0, score))
        return self._score_to_category(score)

    def _calculate_reddit_user_credibility(self, metadata: dict) -> str:
        """Calculate credibility category for Reddit users based on metadata."""
        score = 0.5  # Base score
        
        # Account establishment - karma as indicator of sustained participation
        karma = metadata.get('total_karma', 0)
        if karma > 50000:
            score += 0.15   # Very established account
        elif karma > 10000:
            score += 0.1    # Established account
        elif karma > 1000:
            score += 0.05   # Some establishment
        
        # Account age (establishment indicator)
        account_age_years = metadata.get('account_age_years', 0)
        if account_age_years > 5:
            score += 0.2   # Very established account
        elif account_age_years > 2:
            score += 0.15  # Mature account
        elif account_age_years > 1:
            score += 0.1   # Some history
        
        # Official roles and verification
        if metadata.get('is_employee', False):
            score += 0.3   # High credibility
        if metadata.get('is_mod', False):
            score += 0.2   # Community trust and responsibility
        if metadata.get('verified', False):
            score += 0.15  # Platform verification
        if metadata.get('has_verified_email', False):
            score += 0.05  # Basic verification
        
        # Account integrity (no direct metadata for banned/suspended, would be in account_status)
        account_status = metadata.get('account_status', [])
        if any('suspended' in (status or '').lower() or 'banned' in (status or '').lower() for status in account_status):
            score -= 0.5   # Account with issues
        
        score = max(0.0, min(1.0, score))
        return self._score_to_category(score)

    def _calculate_reddit_user_credibility_fallback(self, text: str) -> str:
        """Fallback method: Calculate credibility category for Reddit users based on text parsing."""
        score = 0.5  # Base score
        
        import re
        
        # Account establishment (NOT engagement) - karma as indicator of sustained participation
        karma_match = re.search(r'total karma: (\d+(?:,\d+)*)', text.lower())
        if karma_match:
            karma = int(karma_match.group(1).replace(',', ''))
            # Use karma as indicator of account establishment, not engagement
            if karma > 50000:
                score += 0.15   # Very established account
            elif karma > 10000:
                score += 0.1    # Established account
            elif karma > 1000:
                score += 0.05   # Some establishment
        
        # Account age (establishment indicator)
        age_match = re.search(r'account age: (\d+) years', text.lower())
        if age_match:
            years = int(age_match.group(1))
            if years > 5:
                score += 0.2   # Very established account
            elif years > 2:
                score += 0.15  # Mature account
            elif years > 1:
                score += 0.1   # Some history
        
        # Official roles and verification
        if 'reddit employee' in text.lower():
            score += 0.3   # High credibility
        if 'moderator' in text.lower():
            score += 0.2   # Community trust and responsibility
        if 'verified' in text.lower():
            score += 0.15  # Platform verification
        if 'email verified' in text.lower():
            score += 0.05  # Basic verification
        
        # Account integrity
        if 'suspended' in text.lower() or 'banned' in text.lower():
            score -= 0.5   # Account with issues
        if 'shadowbanned' in text.lower():
            score -= 0.4   # Hidden account issues
        
        score = max(0.0, min(1.0, score))
        return self._score_to_category(score)

    def _calculate_reddit_subreddit_credibility(self, metadata: dict) -> str:
        """Calculate credibility category for Reddit subreddits based on metadata."""
        score = 0.5  # Base score
        
        # Community establishment (size as indicator of sustained community)
        subscribers = metadata.get('subscribers', 0)
        if subscribers > 1000000:
            score += 0.2   # Very large, established community
        elif subscribers > 100000:
            score += 0.15  # Large community
        elif subscribers > 10000:
            score += 0.1   # Medium community
        elif subscribers > 1000:
            score += 0.05  # Small but viable community
        
        # Community age (establishment indicator)
        created_utc = metadata.get('created_utc')
        if created_utc:
            from datetime import datetime, timezone
            creation_date = datetime.fromtimestamp(created_utc, tz=timezone.utc)
            age_years = (datetime.now(timezone.utc) - creation_date).days // 365
            if age_years > 10:
                score += 0.2   # Very established community
            elif age_years > 5:
                score += 0.15  # Established community
            elif age_years > 2:
                score += 0.1   # Some history
        
        # Moderation and community quality
        if metadata.get('wiki_enabled', False):
            score += 0.05  # Organized community resources
        
        # Community issues
        if metadata.get('quarantine', False):
            score -= 0.4   # Problematic content
        if metadata.get('over_18', False):
            score -= 0.15  # May be less reliable for fact-checking
        
        # Community type
        subreddit_type = metadata.get('subreddit_type', 'public')
        if subreddit_type == 'private':
            score -= 0.1   # Less transparent
        
        # Community info analysis
        community_info = metadata.get('community_info', [])
        if any('quarantined' in info.lower() for info in community_info):
            score -= 0.4
        
        score = max(0.0, min(1.0, score))
        return self._score_to_category(score)

    def _calculate_reddit_subreddit_credibility_fallback(self, text: str) -> str:
        """Fallback method: Calculate credibility category for Reddit subreddits based on text analysis."""
        score = 0.5  # Base score
        
        import re
        
        # Community establishment (size as indicator of sustained community)
        sub_match = re.search(r'subscribers: (\d+(?:,\d+)*)', text.lower())
        if sub_match:
            subscribers = int(sub_match.group(1).replace(',', ''))
            # Larger communities have more established moderation and norms
            if subscribers > 1000000:
                score += 0.2   # Very large, established community
            elif subscribers > 100000:
                score += 0.15  # Large community
            elif subscribers > 10000:
                score += 0.1   # Medium community
            elif subscribers > 1000:
                score += 0.05  # Small but viable community
        
        # Community age (establishment indicator)
        age_match = re.search(r'created: \w+ \d{4} \((\d+) years ago\)', text.lower())
        if age_match:
            years = int(age_match.group(1))
            if years > 10:
                score += 0.2   # Very established community
            elif years > 5:
                score += 0.15  # Established community
            elif years > 2:
                score += 0.1   # Some history
        
        # Moderation and community quality
        if 'active moderation' in text.lower() or 'well moderated' in text.lower():
            score += 0.15  # Quality moderation
        if 'rules enforced' in text.lower():
            score += 0.1   # Clear governance
        if 'wiki enabled' in text.lower():
            score += 0.05  # Organized community resources
        
        # Community issues
        if 'quarantined' in text.lower():
            score -= 0.4   # Problematic content
        if 'banned' in text.lower():
            score -= 0.5   # Severe issues
        if 'nsfw' in text.lower():
            score -= 0.15  # May be less reliable for fact-checking
        if 'controversial' in text.lower():
            score -= 0.1   # May have biased content
        
        # Official or educational communities
        if any(official in text.lower() for official in ['official', 'university', 'academic', 'research']):
            score += 0.25  # Institutional backing
        
        score = max(0.0, min(1.0, score))
        return self._score_to_category(score)

    def _run_async_in_thread(self, coro):
        """Run async coroutine in a separate thread with its own event loop."""
        import concurrent.futures
        import threading
        
        def run_in_thread():
            # Create new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(coro)
                # Ensure session is properly closed
                if hasattr(self, 'session') and self.session and not self.session.closed:
                    new_loop.run_until_complete(self.session.close())
                return result
            finally:
                new_loop.close()
        
        # Use ThreadPoolExecutor to run the async function
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result(timeout=60)  # 60 second timeout

    def get_sync(self, url: str, claim: str) -> Optional[WebSource]:
        """
        Synchronously retrieves and analyzes content from a Reddit URL.
        This is a convenience wrapper around the async get method.
        """
        return self._run_async_in_thread(self.get(url, claim))

    def get_post(self, url: str, claim: str = "") -> Optional[WebSource]:
        """
        Retrieves a post from a Reddit URL and returns it as a WebSource.
        """
        return self._run_async_in_thread(self.get(url, claim))

    def get_user(self, url: str, claim: str = "") -> Optional[WebSource]:
        """
        Retrieves a user profile from a Reddit URL and returns it as a WebSource.
        """
        return self._run_async_in_thread(self.get(url, claim))

    def get_subreddit(self, url: str, claim: str = "") -> Optional[WebSource]:
        """
        Retrieves a subreddit from a Reddit URL and returns it as a WebSource.
        """
        return self._run_async_in_thread(self.get(url, claim))

    async def close(self):
        """Closes the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        """Ensure the aiohttp session is closed when the object is destroyed."""
        if self.session and not self.session.closed:
            try:
                # Use the threading approach to avoid event loop issues
                self._run_async_in_thread(self.session.close())
            except Exception:
                # If we can't close properly, at least try to avoid warnings
                pass


# Create a singleton instance for easy import
reddit = Reddit()
