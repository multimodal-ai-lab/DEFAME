b"""
Credibility scoring for social media content.
Analyzes posts and comments to calculate credibility scores and format content.
"""
from typing import Dict, Any, List, Optional, Tuple
import time

from defame.common.logger import logger

# Import stance detection
try:
    from defame.analysis.stance import detect_comments_stance_batch, StanceLabel
    STANCE_AVAILABLE = True
except ImportError:
    STANCE_AVAILABLE = False
    logger.warning("Stance detection not available for credibility scoring")


def calculate_reddit_credibility(metadata: Dict[str, Any], claim: str = "", post_content: str = "") -> Tuple[str, str]:
    """
    Calculate credibility score for a Reddit post.
    Returns (category, explanation).
    
    Args:
        metadata: Post metadata including author, engagement, comments, etc.
        claim: The claim being fact-checked (for stance detection)
        post_content: The post content text (for stance detection context)
    
    Returns:
        Tuple of (category, explanation)
    """
    logger.info("=" * 60)
    logger.info("CREDIBILITY: Calculating Reddit Post Credibility")
    logger.info("=" * 60)
    
    # Log available metadata for debugging
    logger.info(f"📊 Available metadata keys: {list(metadata.keys())}")
    logger.info(f"📊 Author: {metadata.get('author', 'N/A')}")
    logger.info(f"📊 Author karma: {metadata.get('author_total_karma', 'N/A')}")
    logger.info(f"📊 Author verified: {metadata.get('author_verified', 'N/A')}")
    logger.info(f"📊 Author is mod: {metadata.get('author_is_mod', 'N/A')}")
    logger.info(f"📊 Post score: {metadata.get('score', 'N/A')}")
    logger.info(f"📊 Upvotes: {metadata.get('upvotes', 'N/A')}")
    logger.info(f"📊 Number of comments: {metadata.get('num_comments', 'N/A')}")
    
    # Calculate author credibility (0-60 points, 60% of total)
    author_score, author_explanation = _calculate_reddit_author_score(metadata)
    logger.info(f"Author credibility: {author_score}/60 points (60% of total)")
    logger.info(f"Author factors: {', '.join(author_explanation) if author_explanation else 'None'}")
    
    # Calculate post quality (0-20 points, 20% of total)
    post_score, post_explanation = _calculate_reddit_post_score(metadata)
    logger.info(f"Post quality: {post_score}/20 points (20% of total)")
    logger.info(f"Post factors: {', '.join(post_explanation) if post_explanation else 'None'}")
    
    # Combine author and post scores
    score = author_score + post_score
    explanation_parts = author_explanation + post_explanation
    
    # Add weighted comment stance (-20 to +20 points, 20% of total)
    comments_data = _extract_reddit_comments(metadata)
    
    if comments_data and claim and STANCE_AVAILABLE:
        try:
            logger.info(f"Analyzing {len(comments_data)} comments for stance detection...")
            comment_stance_score = _calculate_weighted_stance(comments_data, claim, post_content)
            stance_modifier = comment_stance_score * 20  # Scale -1.0/+1.0 to -20/+20 points
            
            logger.info(f"Comment stance score: {comment_stance_score:.3f} (-1.0 to +1.0)")
            logger.info(f"Stance modifier: {stance_modifier:+.1f} points (-20 to +20, 20% of total)")
            
            score += stance_modifier
            
            if abs(stance_modifier) > 5:
                if comment_stance_score > 0:
                    explanation_parts.append("Supported by community comments")
                else:
                    explanation_parts.append("Questioned by community comments")
        except Exception as e:
            logger.warning(f"Error calculating Reddit comment stance: {e}")
    else:
        if not comments_data:
            logger.info("No comments available for stance analysis")
        elif not STANCE_AVAILABLE:
            logger.warning("Stance detection not available")
    
    max_score = 100
    category = _score_to_category(score, max_score)
    
    logger.info(f"Final score: {score:.1f}/100 points")
    logger.info(f"Credibility category: {category}")
    logger.info("=" * 60)
    
    # Build explanation
    if explanation_parts:
        explanation = f"{category} ({', '.join(explanation_parts)})"
    else:
        explanation = category
    
    return category, explanation


def calculate_x_credibility(metadata: Dict[str, Any], claim: str = "", post_content: str = "") -> Tuple[str, str]:
    """
    Calculate credibility score for an X/Twitter post.
    Returns (category, explanation).
    
    Args:
        metadata: Post metadata including author, engagement, comments, etc.
        claim: The claim being fact-checked (for stance detection)
        post_content: The post content text (for stance detection context)
    
    Returns:
        Tuple of (category, explanation)
    """
    logger.info("=" * 60)
    logger.info("CREDIBILITY: Calculating X Post Credibility")
    logger.info("=" * 60)
    
    # Start with base author score (max ~80 points = 80% of total weight)
    score, explanation_parts = _calculate_x_author_score(metadata)
    logger.info(f"Base author score: {score}/80 points (80% weight)")
    logger.info(f"Factors: {', '.join(explanation_parts) if explanation_parts else 'None'}")
    
    # Add weighted comment stance if available (max ~20 points = 20% of total weight)
    comments_data = _extract_x_comments(metadata)
    
    if comments_data and claim and STANCE_AVAILABLE:
        try:
            logger.info(f"Analyzing {len(comments_data)} comments for stance detection...")
            comment_stance_score = _calculate_weighted_stance(comments_data, claim, post_content)
            stance_modifier = comment_stance_score * 20  # Scale to ±20 points
            
            logger.info(f"Comment stance score: {comment_stance_score:.3f} (-1.0 to +1.0)")
            logger.info(f"Stance modifier: {stance_modifier:+.1f} points (-20 to +20, 20% of total)")
            
            score += stance_modifier
            
            if abs(stance_modifier) > 5:
                if comment_stance_score > 0:
                    explanation_parts.append("Supported by replies")
                else:
                    explanation_parts.append("Questioned by replies")
        except Exception as e:
            logger.warning(f"Error calculating X comment stance: {e}")
    else:
        if not comments_data:
            logger.info("No comments available for stance analysis")
        elif not STANCE_AVAILABLE:
            logger.warning("Stance detection not available")
    
    max_score = 100
    category = _score_to_category(score, max_score)
    
    logger.info(f"Final score: {score:.1f}/100 points")
    logger.info(f"Credibility category: {category}")
    logger.info("=" * 60)
    
    # Build explanation
    if explanation_parts:
        explanation = f"{category} ({', '.join(explanation_parts)})"
    else:
        explanation = category
    
    return category, explanation


def format_reddit_content(content_str: str, metadata: Dict[str, Any], category: str, explanation: str) -> str:
    """
    Format Reddit post content with credibility rating.
    
    Args:
        content_str: The post content as string
        metadata: Post metadata
        category: Credibility category
        explanation: Credibility explanation
    
    Returns:
        Formatted content string
    """
    # Extract basic metadata
    author = metadata.get("author", "Unknown")
    subreddit = metadata.get("subreddit", "Unknown")
    # Remove "r/" prefix if it exists
    if subreddit.startswith("r/"):
        subreddit = subreddit[2:]
    subreddit_desc = metadata.get("subreddit_description", "")
    timestamp = metadata.get("timestamp") or metadata.get("created_at", "Unknown date")
    url = metadata.get("url", "")
    title = metadata.get("post_title") or metadata.get("title", "")
    
    # Extract engagement metrics
    upvotes = metadata.get("upvotes", 0)
    score = metadata.get("score", 0)
    upvote_ratio = metadata.get("upvote_ratio", 0.0)
    num_comments = metadata.get("num_comments", 0)
    
    # Extract external link info
    url_linked = metadata.get("url_linked", "")
    domain = metadata.get("domain", "")
    is_self = metadata.get("is_self", True)
    
    # Build formatted output
    formatted = f"**Reddit Post by user u/{author}**\n"
    formatted += f"Subreddit: r/{subreddit}\n"
    
    if subreddit_desc:
        formatted += f"**Subreddit description**: {subreddit_desc}\n"
    
    formatted += f"""
Post author: u/{author}
Posted: {timestamp}
URL: {url}
Engagement: {upvotes:,} upvotes, {score:,} score ({upvote_ratio*100:.1f}% upvote ratio)
Comments: {num_comments:,}
"""
    
    # Add credibility rating
    formatted += f"**Credibility Rating**: {explanation}\n\n"
    
    # Add external link if present (before the content)
    if not is_self and url_linked and url_linked != url:
        formatted += f"**External Link**: {url_linked}"
        if domain:
            formatted += f" (Domain: {domain})"
        formatted += "\n\n"
    
    # Add title if we have one
    if title:
        formatted += f"**{title}**\n\n"
    
    # Get the actual post text from metadata (cleaner than parsing content string)
    post_text = metadata.get("post_text", "")
    
    # Extract media references from content
    media_refs = []
    for line in content_str.split('\n'):
        stripped = line.strip()
        if stripped.startswith('<image:') or stripped.startswith('<video:'):
            media_refs.append(stripped)
    
    # Combine post text with media references
    if post_text:
        formatted += post_text
    
    if media_refs:
        formatted += '\n\n' if post_text else ''
        formatted += ' '.join(media_refs)
    
    return formatted


def format_x_content(content_str: str, metadata: Dict[str, Any], category: str, explanation: str, url: str = "") -> str:
    """
    Format X/Twitter post content with credibility rating.
    
    Args:
        content_str: The post content as string
        metadata: Post metadata
        category: Credibility category
        explanation: Credibility explanation
        url: The post URL
    
    Returns:
        Formatted content string
    """
    # Extract basic metadata
    author = metadata.get("author_username", "Unknown")
    author_name = metadata.get("author_name", "Unknown")
    created_at = metadata.get("created_at", "Unknown date")
    
    # Use provided URL or fall back to metadata
    url = url or metadata.get("url", "")
    
    # Extract engagement metrics
    public_metrics = metadata.get("post_public_metrics") or metadata.get("public_metrics", {})
    likes = public_metrics.get("like_count", 0)
    retweets = public_metrics.get("retweet_count", 0)
    replies = public_metrics.get("reply_count", 0)
    views = public_metrics.get("impression_count", 0)
    
    # Extract media references from content
    lines = content_str.split('\n')
    clean_lines = []
    post_text = ""
    
    for line in lines:
        stripped = line.strip()
        # Skip scrapeMM metadata headers
        if stripped.startswith('**X Post by') or stripped.startswith('Author:') or \
           stripped.startswith('Posted:') or stripped.startswith('URL:') or \
           stripped.startswith('Engagement:') or stripped.startswith('Views:'):
            continue
        # Keep media references and actual content
        if stripped.startswith('<image:') or stripped.startswith('<video:') or \
           (stripped and not stripped.startswith('**')):
            clean_lines.append(line)
    
    post_text = '\n'.join(clean_lines).strip()
    
    # Build formatted output
    formatted = f"**X Post by @{author}**\n"
    formatted += f"Author: {author_name}, @{author}\n"
    formatted += f"Posted: {created_at}\n"
    formatted += f"URL: {url}\n"
    formatted += f"Engagement: {likes:,} likes, {retweets:,} retweets, {replies:,} replies\n"
    formatted += f"Views: {views:,}\n"
    formatted += f"**Credibility Rating**: {explanation}\n\n"
    formatted += post_text
    
    return formatted


# Helper functions

def _calculate_reddit_author_score(metadata: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Calculate Reddit author credibility score."""
    score = 0
    explanation = []
    
    # Check if we have any author data at all
    has_author_data = any([
        metadata.get('author_total_karma'),
        metadata.get('author_verified'),
        metadata.get('author_is_employee'),
        metadata.get('author_is_mod'),
        metadata.get('author_account_created'),
        metadata.get('author_is_gold')
    ])
    
    # Special status (highest credibility indicators)
    if metadata.get('author_is_employee', False):
        score += 22
        explanation.append("Reddit employee")
    elif metadata.get('author_is_mod', False):
        score += 15
        explanation.append("Subreddit moderator")
    
    # Account verification
    if metadata.get('author_verified', False) or metadata.get('author_has_verified_email', False):
        score += 4
        explanation.append("Verified account")
    
    # Karma (community trust indicator)
    total_karma = metadata.get('author_total_karma', 0)
    if total_karma > 100000:
        score += 23
        explanation.append("High karma (100K+)")
    elif total_karma > 50000:
        score += 19
        explanation.append("High karma (50K+)")
    elif total_karma > 10000:
        score += 11
        explanation.append("Established contributor (10K+ karma)")
    elif total_karma > 1000:
        score += 4
        explanation.append("Some contribution history (1K+ karma)")
    
    # Account age (sustained presence)
    account_created = metadata.get('author_account_created', 0)
    if account_created > 0:
        account_age_years = (time.time() - account_created) / (365.25 * 24 * 3600)
        if account_age_years > 10:
            score += 15
            explanation.append(f"Long-term account ({int(account_age_years)}+ years)")
        elif account_age_years > 5:
            score += 11
            explanation.append(f"Established account ({int(account_age_years)}+ years)")
        elif account_age_years > 2:
            score += 8
            explanation.append(f"Mature account ({int(account_age_years)}+ years)")
        elif account_age_years > 1:
            score += 4
            explanation.append(f"Active account ({int(account_age_years)}+ year)")
    
    # Premium status
    if metadata.get('author_is_gold', False):
        score += 3
        explanation.append("Reddit Premium user")
    
    # Fallback: If we have no author data, give moderate baseline score
    # This is necessary because scrapeMM often doesn't fetch author details
    # We assume posts on Reddit have passed some basic community filters
    if not has_author_data and metadata.get('author'):
        score += 25  # Moderate baseline (41% of max author score)
        explanation.append("Reddit user (limited author data available)")
        logger.warning(f"⚠️ No author metrics available for u/{metadata.get('author')}. Using moderate baseline score.")
        logger.warning("⚠️ Consider fetching author data separately for more accurate credibility assessment.")
    
    return score, explanation


def _calculate_reddit_post_score(metadata: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Calculate Reddit post quality score."""
    score = 0
    explanation = []
    
    # Content quality indicators
    if metadata.get('is_original_content', False):
        score += 11
        explanation.append("Original content")
    
    # Source attribution
    url_linked = metadata.get('url_linked')
    is_self = metadata.get('is_self', True)
    
    if url_linked and not is_self:
        score += 7
        explanation.append("External source linked")
    
    # Moderator actions
    if metadata.get('locked', False):
        score += 4
        explanation.append("Locked by moderators")
        
    # Content flags
    if metadata.get('over_18', False):
        score -= 4
        explanation.append("NSFW content")
        
    # Flair as credibility indicators
    flair_text = (metadata.get('link_flair_text') or '').lower()
    if any(keyword in flair_text for keyword in ['misleading', 'unverified', 'fake']):
        score -= 15
        explanation.append("Flagged as misleading")
    if any(keyword in flair_text for keyword in ['verified', 'confirmed', 'official']):
        score += 11
        explanation.append("Verified flair")
    
    # Awards as quality indicators
    total_awards = metadata.get('total_awards_received', 0)
    if total_awards >= 5:
        score += 7
        explanation.append("Highly awarded")
    
    return score, explanation


def _calculate_x_author_score(metadata: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Calculate X/Twitter author credibility score."""
    score = 0
    explanation = []
    author_metrics = metadata.get("author_public_metrics", {})
    
    # Verification Status (primary credibility indicator)
    if metadata.get("author_verified"):
        verified_type = metadata.get("author_verified_type", "").lower()
        if verified_type in ["government", "business"]:
            score += 50
            explanation.append("Verified account (gold checkmark)")
        else:
            score += 25
            explanation.append("Verified account (blue checkmark)")
            
    # Account establishment (NOT popularity) - indicates sustained presence
    followers = author_metrics.get("followers_count", 0)
    if followers > 100000:
        score += 15
        explanation.append("Established account (100K+ followers)")
    elif followers > 10000:
        score += 8
        explanation.append("Active account (10K+ followers)")
    elif followers > 1000:
        score += 4
        explanation.append("Some following (1K+ followers)")

    # Account integrity - penalize problematic accounts
    has_negative_flags = False
    
    if metadata.get("author_protected"):
        score -= 10
        explanation.append("Protected account")
        has_negative_flags = True
    if metadata.get("author_withheld"):
        score -= 20
        explanation.append("Withheld content")
        has_negative_flags = True
    if metadata.get("author_suspended"):
        score -= 50
        explanation.append("Suspended account")
        has_negative_flags = True
    
    # Reward accounts with clean integrity
    if not has_negative_flags and score > 0:
        score += 15
        explanation.append("Clean account integrity")
        
    return score, explanation


def _extract_reddit_comments(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and convert Reddit comments from metadata."""
    raw_comments = metadata.get("comments", [])
    if not raw_comments:
        return []
    
    comments_data = []
    for i, comment in enumerate(raw_comments):
        if isinstance(comment, dict):
            comments_data.append({
                'id': comment.get('id', f"comment_{i}"),
                'text': comment.get('body', comment.get('text', str(comment))),
                'author': comment.get('author', 'unknown')
            })
        elif isinstance(comment, str):
            comments_data.append({
                'id': f"comment_{i}",
                'text': comment,
                'author': 'unknown'
            })
    
    return comments_data


def _extract_x_comments(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and convert X/Twitter comments from metadata."""
    raw_comments = metadata.get("comments", [])
    if not raw_comments:
        return []
    
    comments_data = []
    for i, comment in enumerate(raw_comments):
        if isinstance(comment, str):
            # X comments are formatted strings like "Comment by @username (X likes): text"
            # Extract just the text part
            comment_text = comment
            if ':' in comment:
                comment_text = comment.split(':', 1)[1].strip()
            
            comments_data.append({
                'id': f"comment_{i}",
                'text': comment_text,
                'author': 'unknown'
            })
        elif isinstance(comment, dict):
            comments_data.append({
                'id': comment.get('id', f"comment_{i}"),
                'text': comment.get('text', str(comment)),
                'author': comment.get('author', 'unknown')
            })
    
    return comments_data


def _calculate_weighted_stance(comments_data: List[Dict[str, Any]], claim: str, post_content: str) -> float:
    """
    Calculate weighted stance score from comments.
    Returns a score from -1.0 (refute) to +1.0 (support).
    """
    if not comments_data or not STANCE_AVAILABLE:
        return 0.0
    
    try:
        stance_results = detect_comments_stance_batch(comments_data[:50], claim, post_content)
        
        if not stance_results.get("results"):
            return 0.0
        
        total_weight = 0
        weighted_stance_sum = 0
        
        credibility_weights = {
            "entirely credible": 1.0,
            "mostly credible": 0.8,
            "moderately credible": 0.6,
            "somewhat credible": 0.4,
            "entirely uncredible": 0.1
        }
        
        for result in stance_results["results"]:
            try:
                stance_label = result.get("stance")
                confidence = result.get("confidence", "MEDIUM")
                
                comment_credibility = credibility_weights.get("moderately credible", 0.6)
                confidence_multiplier = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}.get(confidence, 0.7)
                weight = comment_credibility * confidence_multiplier
                
                if stance_label == StanceLabel.SUPPORTS:
                    stance_value = 1.0
                elif stance_label == StanceLabel.REFUTES:
                    stance_value = -1.0
                else:
                    stance_value = 0.0
                
                total_weight += weight
                weighted_stance_sum += stance_value * weight
                
            except Exception as e:
                logger.warning(f"Error processing stance result: {e}")
                continue
        
        if total_weight > 0:
            return weighted_stance_sum / total_weight
        return 0.0
            
    except Exception as e:
        logger.error(f"Error in stance calculation: {e}")
        return 0.0


def _score_to_category(score: int, max_score: int) -> str:
    """Map numerical score to credibility category."""
    normalized_score = max(0, min(1, score / max_score))
    
    if normalized_score >= 0.8:
        return "Entirely Credible"
    elif normalized_score >= 0.6:
        return "Mostly Credible"
    elif normalized_score >= 0.4:
        return "Moderately Credible"
    elif normalized_score >= 0.2:
        return "Somewhat Credible"
    else:
        return "Entirely Uncredible"
