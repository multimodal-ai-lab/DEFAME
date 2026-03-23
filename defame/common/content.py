from datetime import datetime
from typing import Optional

from ezmm import MultimodalSequence

from defame.common.label import Label


class Content(MultimodalSequence):
    """The raw content to be interpreted, decomposed, decontextualized, etc. before
    being checked. Media resources are referred in the text """

    claims: Optional[list] = None  # the claims contained in this content
    verdict: Optional[Label] = None  # the overall verdict aggregated from the individual claims
    topic: Optional[str] = None  # a short title-like description of the content's topic

    def __init__(self,
                 *args,
                 author: str = None,
                 date: datetime = None,
                 origin: str = None,  # URL
                 meta_info: str = None,
                 interpretation: str = None,  # Added during claim extraction
                 id: str | int | None = None,  # Used by some benchmarks to identify contents
                 ):
        super().__init__(*args)

        self.author = author
        self.date = date
        self.origin = origin  # URL
        self.meta_info = meta_info
        self.interpretation = interpretation  # added during claim extraction
        self.id = id  # used by some benchmarks and the API backend to identify contents
        self.claims: Optional[list] = None  # the claims contained in this content
        self.verdict: Optional[Label] = None  # the overall verdict aggregated from the individual claims

    def __repr__(self):
        return f"Content(str_len={len(self.__str__())}, id={self.id}, author={self.author})"

    def __str__(self):
        out_string = ""
        if self.author:
            out_string += f"**Author**: {self.author}\n"
        if self.date:
            out_string += f"**Date**: {self.date.strftime('%B %d, %Y')}\n"
        if self.origin:
            out_string += f"**Origin**: {self.origin}\n"
        if self.meta_info:
            out_string += f"**Meta info**: {self.meta_info}\n"
        out_string += f"**Content**: {super().__str__()}"
        return out_string
