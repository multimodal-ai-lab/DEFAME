from abc import ABC
from typing import Any, Optional

import torch

from defame.common import Action, Result, Evidence, MultimediaSnippet, Model


class Tool(ABC):
    """Base class for all tools."""
    name: str
    actions: list[type(Action)]  # (classes of the) available actions this tool offers

    def __init__(self, llm: Model = None, device: str | torch.device = None):
        self.device = device
        self.llm = llm

    def perform(self, action: Action, summarize: bool = True, **kwargs) -> Evidence:
        result = self._perform(action)
        summary = self._summarize(result, **kwargs) if summarize else None
        return Evidence(result, action, summary=summary)

    def _perform(self, action: Action) -> Result:
        """The actual function executing the action."""
        raise NotImplementedError

    def _summarize(self, result: Result, **kwargs) -> Optional[MultimediaSnippet]:
        """Turns the result into an LLM-friendly summary. May use additional
        context for summarization. Returns None iff the result does not contain any
        (potentially) helpful information."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the tool to its initial state (if applicable) and sets all stats to zero."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Returns the tool's usage statistics as a dictionary."""
        return {}


def get_available_actions(tools: list[Tool], available_actions: list[Action]) -> set[type[Action]]:
    actions = set()
    for tool in tools:
        actions.update(tool.actions)
    actions = actions.intersection(set(available_actions))
    return actions
