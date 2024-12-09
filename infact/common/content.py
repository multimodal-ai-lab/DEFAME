from dataclasses import dataclass
from datetime import datetime


@dataclass
class Content:
    """The raw content to be interpreted, decomposed, decontextualized, etc. before
    being checked."""
    text: str

    author: str = None
    date: datetime = None
    origin: str = None  # URL

    interpretation: str = None  # Added during claim extraction

    id_number: int = None  # Used by some benchmarks to identify contents

    def __str__(self):
        return f"Content: \"{self.text}\""
