from dataclasses import dataclass
from datetime import datetime
from infact.common.medium import MultimediaSnippet


@dataclass
class Content(MultimediaSnippet):
    """The raw content to be interpreted, decomposed, decontextualized, etc. before
    being checked. Media resources are referred in the text """

    author: str = None
    date: datetime = None
    origin: str = None  # URL

    interpretation: str = None  # Added during claim extraction

    id_number: int = None  # Used by some benchmarks to identify contents

    def __str__(self):
        return f"Content: {super().__str__()}"
