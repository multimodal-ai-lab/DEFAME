from dataclasses import dataclass
from abc import ABC
from typing import Any


@dataclass
class Action(ABC):
    name: str


@dataclass
class SearchAction(Action):
    api: str
    query: str


@dataclass
class Result:
    action: Action
    result: Any


class Actor:
    def __init__(self):
        pass

    def perform(self, actions: list[Action]) -> list[Result]:
        return [self._perform_single(a) for a in actions]

    def _perform_single(self, action: Action) -> Result:
        if isinstance(action, SearchAction):
            return self._perform_search(action)
        else:
            raise ValueError(f"Action name '{action.name}' unknown.")

    def _perform_search(self, search_action: SearchAction) -> Result:
        raise NotImplementedError()
