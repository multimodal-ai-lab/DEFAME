from datetime import datetime
from typing import Optional
import time

from defame.common import Action, Report, Evidence, logger
from defame.evidence_retrieval.tools import Tool, Searcher


class Actor:
    """Agent that executes given Actions and returns the resulted Evidence."""

    def __init__(self, tools: list[Tool]):
        self.tools = tools

    def perform(self, actions: list[Action], doc: Report = None, summarize: bool = True) -> list[Evidence]:
        # TODO: Parallelize
        all_evidence = []
        for i, action in enumerate(actions):
            assert isinstance(action, Action)
            logger.log(f"[Actor] Starting action {i+1}/{len(actions)}: {type(action).__name__}")
            start_time = time.time()
            try:
                evidence = self._perform_single(action, doc, summarize=summarize)
                elapsed = time.time() - start_time
                logger.log(f"[Actor] Completed action {i+1}/{len(actions)} in {elapsed:.2f}s")
                all_evidence.append(evidence)
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[Actor] Action {i+1}/{len(actions)} failed after {elapsed:.2f}s: {e}")
                raise
        return all_evidence

    def _perform_single(self, action: Action, doc: Report = None, summarize: bool = True) -> Evidence:
        tool = self.get_corresponding_tool_for_action(action)
        logger.log(f"[Actor] Using tool: {tool.name} for action: {action}")
        start_time = time.time()
        try:
            evidence = tool.perform(action, summarize=summarize, doc=doc)
            elapsed = time.time() - start_time
            logger.log(f"[Actor] Tool {tool.name} completed in {elapsed:.2f}s")
            return evidence
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Actor] Tool {tool.name} failed after {elapsed:.2f}s: {e}")
            raise

    def get_corresponding_tool_for_action(self, action: Action) -> Tool:
        for tool in self.tools:
            if type(action) in tool.actions:
                return tool
        raise ValueError(f"No corresponding tool available for Action '{action}'.")

    def reset(self):
        """Resets all tools (if applicable)."""
        for tool in self.tools:
            tool.reset()

    def set_current_claim_id(self, claim_id: str):
        for tool in self.tools:
            tool.set_claim_id(claim_id)

    def get_tool_stats(self):
        return {t.name: t.get_stats() for t in self.tools}

    def _get_searcher(self) -> Optional[Searcher]:
        for tool in self.tools:
            if isinstance(tool, Searcher):
                return tool

    def set_search_date_restriction(self, before: Optional[datetime]):
        searcher = self._get_searcher()
        if searcher is not None:
            searcher.set_time_restriction(before)
