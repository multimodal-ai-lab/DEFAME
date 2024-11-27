from datetime import date
from typing import Optional

from defame.common import Action, Report, Evidence
from defame.tools import Tool, Searcher


class Actor:
    """Agent that executes given Actions and returns the resulted Evidence."""

    def __init__(self, tools: list[Tool]):
        self.tools = tools

    def perform(self, actions: list[Action], doc: Report = None, summarize: bool = True) -> list[Evidence]:
        # TODO: Parallelize
        all_evidence = []
        for action in actions:
            assert isinstance(action, Action)
            all_evidence.append(self._perform_single(action, doc, summarize=summarize))
        return all_evidence

    def _perform_single(self, action: Action, doc: Report = None, summarize: bool = True) -> Evidence:
        tool = self.get_corresponding_tool_for_action(action)
        return tool.perform(action, summarize=summarize, doc=doc)

    def get_corresponding_tool_for_action(self, action: Action) -> Tool:
        for tool in self.tools:
            if type(action) in tool.actions:
                return tool
        raise ValueError(f"No corresponding tool available for Action '{action}'.")

    def reset(self):
        """Resets all tools (if applicable)."""
        for tool in self.tools:
            tool.reset()

    def get_tool_stats(self):
        return {t.name: t.get_stats() for t in self.tools}

    def _get_searcher(self) -> Optional[Searcher]:
        for tool in self.tools:
            if isinstance(tool, Searcher):
                return tool

    def set_search_date_restriction(self, until: date):
        searcher = self._get_searcher()
        if searcher is not None:
            searcher.set_date_restriction(until)
