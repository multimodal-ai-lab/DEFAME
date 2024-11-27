from datetime import datetime

from defame.common.medium import MultimediaSnippet, Medium


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
                 id_number: int = None,  # Used by some benchmarks to identify contents
                 ):
        super().__init__(content=content)
        self.author = author
        self.date = date
        self.origin = origin
        self.meta_info = meta_info
        self.interpretation = interpretation
        self.id_number = id_number

    def __str__(self):
        return f"Content: {super().__str__()}"
