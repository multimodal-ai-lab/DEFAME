from enum import Enum
from typing import Optional, List, Dict, Any
import json

from defame.common.modeling import Model
from defame.common.log import logger


class StanceLabel(Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"
    UNRELATED = "unrelated"


class Stance(Enum):
    SUPPORTING = "supporting"
    REFUTING = "refuting"
    COMMENTING = "commenting"
    UNRELATED = "unrelated"


class StanceDetector:
    """A class to handle stance detection using an LLM."""
    
    def __init__(self, llm: Model):
        self.llm = llm
    
    def detect_stance(self, text_content: str, claim: str) -> Optional[Stance]:
        """
        Detects the stance of a given text towards a specific claim using an LLM.
        """
        prompt = f"""
Analyze the following text and determine its stance towards the claim provided.
The possible stances are: SUPPORTING, REFUTING, COMMENTING, UNRELATED.

- SUPPORTING: The text provides evidence or arguments that directly support the claim.
- REFUTING: The text provides evidence or arguments that directly contradict the claim.
- COMMENTING: The text discusses the claim without taking a clear supportive or refuting stance.
- UNRELATED: The text does not address the claim.

Claim: "{claim}"

Text: "{text_content}"

Based on the text, what is the single most appropriate stance? Respond with only one word from the list of possible stances.
"""
        
        logger.debug(f"Stance detection prompt: {prompt}")
        
        try:
            response = self.llm.generate(prompt)
            logger.info(f"Stance detection response: {response}")
            
            if response and isinstance(response, str):
                stance_text = response.strip().upper()
                detected_stance = Stance(stance_text.lower())
                logger.info(f"Detected stance: {detected_stance.value}")
                return detected_stance
            
            logger.warning("No valid response received from LLM for stance detection")
            return None
        except (ValueError, AttributeError) as e:
            logger.error(f"Error in stance detection: {e}")
            return None

    def detect_comments_stance_batch(self, comments_data: List[Dict[str, Any]], claim: str, post_content: str) -> Dict[str, Any]:
        """
        Detects the stance of multiple comments towards a specific claim using batch processing.
        
        Args:
            comments_data: List of comment dictionaries with 'id', 'text', and other metadata
            claim: The claim to analyze stance against
            
        Returns:
            Dictionary containing the results with comment stances and reasoning
        """
        # Format comments for JSON input
        comments_json = {
            "claim": claim,
            "comments": []
        }
        
        for i, comment in enumerate(comments_data):
            comment_entry = {
                "id": comment.get('id', f"comment_{i}"),
                "text": comment.get('text', ''),
                "author": comment.get('author', 'unknown')
            }
            comments_json["comments"].append(comment_entry)
        
        # Create the prompt
        prompt = f"""
You are analyzing social media comments to determine their stance towards a specific post.
The claim to check is: "{claim}"
the related post is: "{post_content}"
For each comment, determine its stance and provide a brief reason. The possible stances are:
- SUPPORTS: The comment provides evidence or arguments that directly support the claim
- REFUTES: The comment provides evidence or arguments that directly contradict the claim  
- NEUTRAL: The comment discusses the claim without taking a clear supportive or refuting stance
- UNRELATED: The comment does not address the claim at all

Input JSON:
{json.dumps(comments_json, indent=2)}

Please respond with a JSON object in exactly this format:
{{
    "claim": "the original claim",
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
        
        logger.info(f"Batch stance detection for {len(comments_data)} comments")
        logger.info(f"Batch stance prompt:\n{prompt}")
        
        try:
            # Make sure we're only passing text to the LLM
            response = self.llm.generate(prompt)
            logger.info(f"Batch stance response received: {len(response) if response else 0} characters")
            logger.info(f"Batch stance raw response:\n{response}")
            
            if response and isinstance(response, str):
                # Parse the JSON response
                try:
                    result = json.loads(response.strip())
                    
                    # Validate the response structure
                    if "results" in result and isinstance(result["results"], list):
                        # Convert stance strings to StanceLabel enums
                        for item in result["results"]:
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
                                except ValueError:
                                    logger.warning(f"Invalid stance value: {stance_str}, defaulting to NEUTRAL")
                                    item["stance"] = StanceLabel.NEUTRAL
                        
                        logger.info(f"Successfully processed {len(result['results'])} comment stances")
                        logger.info(f"Final processed result: {result}")
                        return result
                    else:
                        logger.error("Invalid JSON structure in response")
                        return {"results": []}
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response: {response}")
                    return {"results": []}
            
            logger.warning("No valid response received from LLM for batch stance detection")
            return {"results": []}
            
        except AttributeError as e:
            if "accepts_videos" in str(e):
                logger.warning(f"Model compatibility issue (ignoring): {e}")
                # Fallback: try with a simpler generate call
                try:
                    response = self.llm.generate(prompt)
                    logger.debug(f"Batch stance response received: {len(response) if response else 0} characters")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    return {"results": []}
            else:
                logger.error(f"Unexpected attribute error in batch stance detection: {e}")
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


def detect_stance(text_content: str, claim: str) -> Optional[Stance]:
    """
    Detects the stance of a given text towards a specific claim using an LLM.
    This function uses the global stance detector instance.
    """
    if _stance_detector is None:
        raise RuntimeError("Stance detector not initialized. Call init_stance_detector() first.")
    
    return _stance_detector.detect_stance(text_content, claim)


def detect_comments_stance_batch(comments_data: List[Dict[str, Any]], claim: str, post_content: str) -> Dict[str, Any]:
    """
    Detects the stance of multiple comments towards a specific claim using batch processing.
    This function uses the global stance detector instance.
    
    Args:
        comments_data: List of comment dictionaries with 'id', 'text', and other metadata
        claim: The claim to analyze stance against
        
    Returns:
        Dictionary containing the results with comment stances and reasoning
    """
    if _stance_detector is None:
        raise RuntimeError("Stance detector not initialized. Call init_stance_detector() first.")

    return _stance_detector.detect_comments_stance_batch(comments_data, claim, post_content)