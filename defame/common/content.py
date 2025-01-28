from datetime import datetime
from typing import Optional

from defame.common.medium import MultimediaSnippet, Medium
from defame.common.label import Label


class Content(MultimediaSnippet):
    """The raw content to be interpreted, decomposed, decontextualized, etc. before
    being checked. Media resources are referred in the text """

    def __init__(self,
                 content: str | list[str | Medium],
                 author: str = None,
                 date: datetime = None,
                 origin: str = None,  # URL
                 meta_info: str = None,
                 interpretation: str = None,  # Added during claim extraction
                 identifier: int | str = None,  # Used by some benchmarks to identify contents
                 ):
        super().__init__(data=content)
        self.author = author
        self.date = date
        self.origin = origin
        self.meta_info = meta_info
        self.interpretation = interpretation
        self.id: str = str(identifier)
        self.claims: Optional[list] = None # the claims contained in this content
        self.verdict: Optional[Label] = None  # the overall verdict aggregated from the individual claims

    def __str__(self):
        return f"Content: {super().__str__()}"

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('claims', None)
        return state
