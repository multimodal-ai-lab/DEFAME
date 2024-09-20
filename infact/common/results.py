from abc import ABC


class Result(ABC):
    """Detailed and raw information piece retrieved by performing an action.
    May contain any data but must implement the string function to enable
    LLMs process this result."""

    def __str__(self):
        """LLM-friendly string representation of the result in Markdown format."""
        raise NotImplementedError
