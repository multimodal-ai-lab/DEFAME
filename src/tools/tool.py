from abc import ABC

from common.action import Action
from common.results import Result


class Tool(ABC):
    """Base class for all tools."""
    name: str
    actions: list[type(Action)]  # (classes of the) available actions this tool offers

    def perform(self, action: Action) -> list[Result]:
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the tool to its initial state (if applicable)."""
        pass


def get_available_actions(tools: list[Tool]) -> set[type[Action]]:
    actions = set()
    for tool in tools:
        actions.update(tool.actions)
    return actions
