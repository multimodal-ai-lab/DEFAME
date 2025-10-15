from datetime import datetime
from typing import Optional

from defame.common import Action, Report, Evidence, logger
from defame.evidence_retrieval.tools import Tool, Searcher


class Actor:
    """Agent that executes given Actions and returns the resulted Evidence."""

    def __init__(self, tools: list[Tool]):
        self.tools = tools
        
        # Log available tools and their actions
        logger.info(f"🎭 Actor initialized with {len(tools)} tool(s):")
        for tool in tools:
            action_names = [a.name if hasattr(a, 'name') else str(a) for a in tool.actions]
            logger.info(f"   - {tool.name}: {', '.join(action_names)}")

    def perform(self, actions: list[Action], doc: Report = None, summarize: bool = True) -> list[Evidence]:
        # TODO: Parallelize
        all_evidence = []
        logger.info(f"🎬 Actor executing {len(actions)} action(s)...")
        for i, action in enumerate(actions, 1):
            assert isinstance(action, Action)
            logger.info(f"   Executing action {i}/{len(actions)}: {type(action).__name__}")
            all_evidence.append(self._perform_single(action, doc, summarize=summarize))
        return all_evidence

    def _perform_single(self, action: Action, doc: Report = None, summarize: bool = True) -> Evidence:
        tool = self.get_corresponding_tool_for_action(action)
        # Extract claim text from doc if available
        claim_text = str(doc.claim) if doc and hasattr(doc, 'claim') else ''
        return tool.perform(action, summarize=summarize, doc=doc, claim=claim_text)

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
