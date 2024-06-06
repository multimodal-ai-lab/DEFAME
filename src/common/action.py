from abc import ABC
from dataclasses import dataclass


@dataclass
class Action(ABC):
    pass


@dataclass
class SearchAction(Action):
    api: str
    query: str

    def __str__(self):
        return f"{self.api}: {self.query}"
