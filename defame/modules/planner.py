from typing import Collection

import pyparsing as pp

from defame.common.action import (Action)
from defame.evidence_retrieval.tools import IMAGE_ACTIONS
from defame.common import logger, Report, Model
from defame.prompts.prompts import PlanPrompt


class Planner:
    """Chooses the next actions to perform based on the current knowledge as contained
    in the FC document."""

    def __init__(self,
                 valid_actions: Collection[type[Action]],
                 llm: Model,
                 extra_rules: str):
        self.valid_actions = valid_actions
        self.llm = llm
        self.max_attempts = 5
        self.extra_rules = extra_rules

    def get_available_actions(self, doc: Report):
        available_actions = list(self.valid_actions.copy())
        completed_actions = set(type(a) for a in doc.get_all_actions())

        # Remove already completed actions
        available_actions = [a for a in available_actions if a not in completed_actions]

        # Add image-specific actions if claim has images
        if doc.claim.has_images():
            for action in IMAGE_ACTIONS:
                if action not in completed_actions and action not in available_actions:
                    available_actions.append(action)

        # Add SearchX if X/Twitter URLs are detected in the claim OR in retrieved evidence
        x_urls_found = []
        all_evidence = doc.get_all_evidence()
        
        # Check original claim for X URLs
        if any(domain in str(doc.claim) for domain in ["twitter.com", "x.com"]):
            x_urls_found.extend(self._extract_x_urls(str(doc.claim)))
        
        # Check all evidence for X URLs
        for evidence in all_evidence:
            evidence_text = str(evidence.raw) if evidence.raw else ""
            if any(domain in evidence_text for domain in ["twitter.com", "x.com"]):
                x_urls_found.extend(self._extract_x_urls(evidence_text))
        
        # Store X URLs for use in prompt
        self.x_urls_for_prompt = list(set(x_urls_found))
        if self.x_urls_for_prompt:
            logger.info(f"Found X URLs for prompt: {self.x_urls_for_prompt}")
        
        # Add SearchX actions for each unique X URL found
        if x_urls_found:
            from defame.evidence_retrieval.tools.x import SearchX
            
            # Get already completed SearchX URLs
            completed_x_urls = set()
            for action in doc.get_all_actions():
                if type(action).__name__ == 'SearchX':
                    completed_x_urls.add(action.url)
            
            # Add SearchX for new URLs
            new_x_urls = set(x_urls_found) - completed_x_urls
            for url in new_x_urls:
                if SearchX not in available_actions:  # Add the class, not instance
                    available_actions.append(SearchX)
                    logger.info(f"Added SearchX to available actions due to X/Twitter URL found: {url}")
                    break  # Only add the action class once

        # Add SearchReddit if Reddit URLs are detected in the claim OR in retrieved evidence
        reddit_urls_found = []
        
        # Check original claim for Reddit URLs
        if "reddit.com" in str(doc.claim):
            reddit_urls_found.extend(self._extract_reddit_urls(str(doc.claim)))
            logger.info(f"🔍 URL DEBUG: Found Reddit URLs in claim: {len(reddit_urls_found)}")
        
        # Check all evidence for Reddit URLs  
        logger.info(f"🔍 URL DEBUG: Checking {len(all_evidence)} evidence items for Reddit URLs")
        for i, evidence in enumerate(all_evidence):
            evidence_text = str(evidence.raw) if evidence.raw else ""
            logger.info(f"🔍 URL DEBUG: Evidence {i+1} type: {type(evidence.raw)}, has reddit.com: {'reddit.com' in evidence_text}")
            if evidence.raw and hasattr(evidence.raw, 'sources'):
                logger.info(f"🔍 URL DEBUG: Evidence {i+1} has {len(evidence.raw.sources)} sources")
                for j, source in enumerate(evidence.raw.sources[:3]):
                    logger.info(f"🔍 URL DEBUG: Source {j+1}: {source.reference}")
            
            if "reddit.com" in evidence_text:
                found_urls = self._extract_reddit_urls(evidence_text)
                reddit_urls_found.extend(found_urls)
                logger.info(f"🔍 URL DEBUG: Found {len(found_urls)} Reddit URLs in evidence text")
            # Also check the reference field for URLs
            if hasattr(evidence, 'reference') and "reddit.com" in str(evidence.reference):
                reddit_urls_found.append(str(evidence.reference))
                logger.info(f"🔍 URL DEBUG: Found Reddit URL in evidence reference: {evidence.reference}")
        
        # Store Reddit URLs for use in prompt
        self.reddit_urls_for_prompt = list(set(reddit_urls_found))
        if self.reddit_urls_for_prompt:
            logger.info(f"Found Reddit URLs for prompt: {self.reddit_urls_for_prompt}")
        
        # Add SearchReddit actions for each unique Reddit URL found
        if reddit_urls_found:
            from defame.evidence_retrieval.tools.reddit import SearchReddit
            
            # Get already completed SearchReddit URLs
            completed_reddit_urls = set()
            for action in doc.get_all_actions():
                if type(action).__name__ == 'SearchReddit':
                    completed_reddit_urls.add(action.url)
            
            # Add SearchReddit for new URLs
            new_reddit_urls = set(reddit_urls_found) - completed_reddit_urls
            for url in new_reddit_urls:
                if SearchReddit not in available_actions:  # Add the class, not instance
                    available_actions.append(SearchReddit)
                    logger.info(f"Added SearchReddit to available actions due to Reddit URL found: {url}")
                    break  # Only add the action class once

        return available_actions

    def _extract_x_urls(self, text: str) -> list[str]:
        """Extract X/Twitter URLs from text."""
        import re
        
        # Pattern to match X/Twitter URLs
        pattern = r'https?://(?:www\.)?(?:twitter\.com|x\.com)/[^\s<>"]*'
        urls = re.findall(pattern, text, re.IGNORECASE)
        
        # Clean URLs (remove trailing punctuation)
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation that might not be part of the URL
            url = re.sub(r'[.,;!?"\')}\]]*$', '', url)
            cleaned_urls.append(url)
        
        return list(set(cleaned_urls))  # Return unique URLs

    def _extract_reddit_urls(self, text: str) -> list[str]:
        """Extract Reddit URLs from text."""
        import re
        
        # Pattern to match Reddit URLs
        pattern = r'https?://(?:www\.)?reddit\.com/r/[^\s<>"]*'
        urls = re.findall(pattern, text, re.IGNORECASE)
        
        # Clean URLs (remove trailing punctuation)
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation that might not be part of the URL
            url = re.sub(r'[.,;!?"\')}\]]*$', '', url)
            cleaned_urls.append(url)
        
        return list(set(cleaned_urls))  # Return unique URLs

    def plan_next_actions(self, doc: Report, all_actions=False) -> tuple[list[Action], str]:
        # Use the refined available actions instead of all valid actions
        available_actions = self.get_available_actions(doc)
        
        if not available_actions:
            logger.warning("No available actions to plan.")
            return [], ""

        # Get specific URLs to use for social media actions
        reddit_urls = getattr(self, 'reddit_urls_for_prompt', [])
        x_urls = getattr(self, 'x_urls_for_prompt', [])

        # Debug logging
        all_actions = doc.get_all_actions()  
        logger.info(f"🔍 PLANNER DEBUG: Planning iteration with {len(all_actions)} completed actions")
        logger.info(f"🔍 PLANNER DEBUG: Planning with {len(available_actions)} available actions")
        logger.info(f"🔍 PLANNER DEBUG: Available action types: {[a.__name__ for a in available_actions]}")
        logger.info(f"🔍 PLANNER DEBUG: Reddit URLs for prompt: {len(reddit_urls)}")
        if reddit_urls:
            for i, url in enumerate(reddit_urls[:3]):
                logger.info(f"🔍 PLANNER DEBUG: Reddit URL {i+1}: {url[:80]}...")

        prompt = PlanPrompt(doc, available_actions, self.extra_rules, all_actions, 
                           reddit_urls=reddit_urls, x_urls=x_urls)
        n_attempts = 0

        while n_attempts < self.max_attempts:
            n_attempts += 1

            response = self.llm.generate(prompt)
            if response is None:
                logger.warning("No new actions were found.")
                return [], ""

            actions = response["actions"]
            reasoning = response["reasoning"]
            
            # Debug logging for generated actions
            logger.info(f"🔍 PLANNER DEBUG: LLM generated {len(actions)} actions")
            for i, action in enumerate(actions):
                logger.info(f"🔍 PLANNER DEBUG: Action {i+1}: {type(action).__name__} - {action}")

            # Remove actions that have been performed before
            performed_actions = doc.get_all_actions()
            actions = [action for action in actions if action not in performed_actions]

            if len(actions) > 0:
                return actions, reasoning
            else:
                performed_actions_str = ", ".join(str(obj) for obj in performed_actions)
                logger.warning(f'No new actions were found in this response:\n{response["response"]} and performed actions: {performed_actions_str}')
                return [], ""


def _process_answer(answer: str) -> str:
    reasoning = answer.split("NEXT_ACTIONS:")[0].strip()
    return reasoning.replace("REASONING:", "").strip()


def _extract_arguments(arguments_str: str) -> list[str]:
    """Separates the arguments_str at all commas that are not enclosed by quotes."""
    ppc = pp.pyparsing_common

    # Setup parser which separates at each comma not enclosed by a quote
    csl = ppc.comma_separated_list()

    # Parse the string using the created parser
    parsed = csl.parse_string(arguments_str)

    # Remove whitespaces and split into arguments list
    return [str.strip(value) for value in parsed]
