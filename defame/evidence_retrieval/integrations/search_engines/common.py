from dataclasses import dataclass

from defame.common import Results
from defame.common.misc import WebSource


@dataclass
class SearchResults(Results):
    """A collection of web sources."""
    sources: list[WebSource]

    def __str__(self):
        return "**Web Search Result**\n\n" +"\n\n".join(map(str, self.sources))


@dataclass
class ReverseSearchResult(SearchResults):
    """Ships with additional object detection information next to the list of web sources."""
    entities: dict[str, float]  # mapping between entity description and confidence score
    best_guess_labels: list[str]

    def __str__(self):
        string_repr = "Google Vision's outputs"
        if self.entities:
            string_repr += f"\nIdentified web entities (confidence in parenthesis):\n"
            string_repr += "\n".join(f"{description} ({confidence * 100:.0f} %)"
                                     for description, confidence in self.entities.items())
        if self.best_guess_labels:
            string_repr += (f"\nBest guess about the topic of "
                           f"the image is {', '.join(self.best_guess_labels)}.\n Exact image matches found at:")
        return "**Reverse Search Result** The exact image was found in the following sources:\n\n" + "\n\n".join(map(str, self.sources))
