from abc import ABC


class Results(ABC):
    """The raw output of a tool after performing a specific action.
    May contain any data. Must implement the string function to enable
    LLMs process this result."""

    def __str__(self):
        """LLM-friendly string representation of the result in Markdown format."""
        raise NotImplementedError
