from abc import ABC


class Action(ABC):
    """Executed by the actor. Performing an Action yields Evidence."""
    name: str
    description: str
    how_to: str
    format: str
    is_multimodal: bool = False

    def __str__(self):
        return f"{self.name}: {self.description}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
