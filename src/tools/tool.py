from abc import ABC

import torch

from src.common.action import Action
from src.common.results import Result
from src.eval.logger import EvaluationLogger


class Tool(ABC):
    """Base class for all tools."""
    name: str
    actions: list[type(Action)]  # (classes of the) available actions this tool offers

    def __init__(self, logger: EvaluationLogger = None, device: str | torch.device = None):
        self.logger = logger or EvaluationLogger()
        self.device = device

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
