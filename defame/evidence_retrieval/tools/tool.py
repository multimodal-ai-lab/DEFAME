from abc import ABC
from typing import Any, Optional

import torch
from ezmm import MultimodalSequence

from defame.common import Action, Results, Evidence, Model


class Tool(ABC):
    """Base class for all tools. Tools leverage integrations to retrieve evidence."""
    name: str
    actions: list[type(Action)]  # (classes of the) available actions this tool offers

    def __init__(self, llm: Model = None, device: str | torch.device = None):
        self.device = device
        self.llm = llm

        self.current_claim_id: Optional[str] = None  # used by few tools to adjust claim-specific behavior

    def perform(self, action: Action, summarize: bool = True, **kwargs) -> Evidence:
        assert type(action) in self.actions, f"Forbidden action: {action}"
        result = self._perform(action)
        summary = self._summarize(result, **kwargs) if summarize else None
        return Evidence(result, action, takeaways=summary)

    def _perform(self, action: Action) -> Results:
        """The actual function executing the action."""
        raise NotImplementedError

    def _summarize(self, result: Results, **kwargs) -> Optional[MultimodalSequence]:
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

    def set_claim_id(self, claim_id: str):
        self.current_claim_id = claim_id


def get_available_actions(tools: list[Tool], available_actions: Optional[list[Action]]) -> set[type[Action]]:
    actions = set()
    for tool in tools:
        actions.update(tool.actions)
    if available_actions is not None:
        actions = actions.intersection(set(available_actions))
    return actions
