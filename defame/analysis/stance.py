from enum import Enum
from typing import Optional, List, Dict, Any
import json

from defame.common.modeling import Model
from defame.common.logger import logger


class StanceLabel(Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"
    UNRELATED = "unrelated"


class StanceDetector:
    """A class to handle batch stance detection using an LLM."""
    
    def __init__(self, llm: Model):
        self.llm = llm

    def detect_comments_stance_batch(self, comments_data: List[Dict[str, Any]], claim: str, post_content: str) -> Dict[str, Any]:
        """
        Detects the stance of multiple comments towards a specific post using batch processing.
        
        Args:
            comments_data: List of comment dictionaries with 'id', 'text', and other metadata
            claim: The claim to analyze stance against (kept for backwards compatibility but not used)
            post_content: The content of the original post
            
        Returns:
            Dictionary containing the results with comment stances and reasoning
        """
        logger.info(f"=== Starting batch stance detection ===")
        logger.info(f"Post content: {post_content[:100]}{'...' if len(post_content) > 100 else ''}")
        logger.info(f"Number of comments to analyze: {len(comments_data)}")
        
        # Format comments for JSON input
        comments_json = {
            "post": post_content,
            "comments": []
        }
        
        for i, comment in enumerate(comments_data):
            comment_entry = {
                "id": comment.get('id', f"comment_{i}"),
                "text": comment.get('text', ''),
                "author": comment.get('author', 'unknown')
            }
            comments_json["comments"].append(comment_entry)
            logger.debug(f"Comment {i+1}: {comment_entry['text'][:80]}{'...' if len(comment_entry['text']) > 80 else ''}")
        
        # Create the prompt
        prompt = f"""
You are analyzing social media comments to determine their stance towards a specific post.

The post content is: "{post_content}"

For each comment, determine its stance towards the POST and provide a brief reason. The possible stances are:
- SUPPORTS: The comment agrees with, validates, or provides evidence that supports the post's message
- REFUTES: The comment disagrees with, challenges, or provides evidence that contradicts the post's message
- NEUTRAL: The comment discusses the post without taking a clear supportive or refuting stance
- UNRELATED: The comment does not address the post's content at all

Input JSON:
{json.dumps(comments_json, indent=2)}

Please respond with a JSON object in exactly this format:
{{
    "post": "the original post content",
    "results": [
        {{
            "comment_id": "comment_1",
            "stance": "SUPPORTS|REFUTES|NEUTRAL|UNRELATED",
            "reason": "Brief explanation of why this stance was chosen",
            "confidence": "HIGH|MEDIUM|LOW"
        }}
    ]
}}

Ensure your response is valid JSON only, no additional text.
"""
        
        logger.info(f"=== Stance Detection Prompt ===")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Prompt content:\n{prompt}")
        logger.info(f"=== End of Prompt ===")
        
        logger.info(f"Sending batch stance detection request to LLM...")
        
        try:
            # Make sure we're only passing text to the LLM
            response = self.llm.generate(prompt)
            logger.info(f"Received response from LLM: {len(response) if response else 0} characters")
            logger.info(f"=== LLM Response ===")
            logger.info(f"{response}")
            logger.info(f"=== End of Response ===")
            
            if response and isinstance(response, str):
                # Parse the JSON response
                try:
                    result = json.loads(response.strip())
                    
                    # Validate the response structure
                    if "results" in result and isinstance(result["results"], list):
                        logger.info(f"Successfully parsed JSON with {len(result['results'])} results")
                        
                        # Convert stance strings to StanceLabel enums
                        for idx, item in enumerate(result["results"]):
                            if "stance" in item:
                                stance_str = item["stance"].upper()
                                try:
                                    # Map to our StanceLabel enum
                                    if stance_str == "SUPPORTS":
                                        item["stance"] = StanceLabel.SUPPORTS
                                    elif stance_str == "REFUTES":
                                        item["stance"] = StanceLabel.REFUTES
                                    elif stance_str == "NEUTRAL":
                                        item["stance"] = StanceLabel.NEUTRAL
                                    elif stance_str == "UNRELATED":
                                        item["stance"] = StanceLabel.UNRELATED
                                    else:
                                        logger.warning(f"Unknown stance: {stance_str}, defaulting to NEUTRAL")
                                        item["stance"] = StanceLabel.NEUTRAL
                                    
                                    logger.info(f"Comment {idx+1} ({item.get('comment_id', 'unknown')}): {item['stance'].value.upper()} - {item.get('reason', 'No reason')[:60]}...")
                                    
                                except ValueError:
                                    logger.warning(f"Invalid stance value: {stance_str}, defaulting to NEUTRAL")
                                    item["stance"] = StanceLabel.NEUTRAL
                        
                        logger.info(f"=== Batch stance detection complete: {len(result['results'])} comments processed ===")
                        return result
                    else:
                        logger.error("Invalid JSON structure in response")
                        return {"results": []}
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    return {"results": []}
            
            logger.warning("No valid response received from LLM for batch stance detection")
            return {"results": []}
            
        except Exception as e:
            logger.error(f"Error in batch stance detection: {e}")
            return {"results": []}


# Global stance detector instance - will be initialized when needed
_stance_detector: Optional[StanceDetector] = None


def init_stance_detector(llm: Model):
    """Initialize the global stance detector with an LLM."""
    global _stance_detector
    _stance_detector = StanceDetector(llm)
    logger.info("Stance detector initialized")


def detect_comments_stance_batch(comments_data: List[Dict[str, Any]], claim: str, post_content: str) -> Dict[str, Any]:
    """
    Detects the stance of multiple comments towards a specific post using batch processing.
    This function uses the global stance detector instance.
    
    Args:
        comments_data: List of comment dictionaries with 'id', 'text', and other metadata
        claim: The claim to analyze stance against (kept for backwards compatibility but not used)
        post_content: The content of the original post
        
    Returns:
        Dictionary containing the results with comment stances and reasoning
    """
    if _stance_detector is None:
        raise RuntimeError("Stance detector not initialized. Call init_stance_detector() first.")

    return _stance_detector.detect_comments_stance_batch(comments_data, claim, post_content)
