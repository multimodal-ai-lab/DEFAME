from typing import Optional

from defame.common.content import Content
from defame.common.medium import MultimediaSnippet, Medium


class Claim(MultimediaSnippet):
    original_context: Content

    def __init__(self, text: str | list[str | Medium],
                 original_context: Optional[Content] = None,
                 identifier: int | str = None):
        super().__init__(text)
        if original_context is None:
            original_context = Content(text)
        self.original_context = original_context  # rhe input the claim was extracted from
        self.scope: (int, int) = None  # TODO: the range in the original context's text that belongs to this claim
        self.id: str = str(identifier)

    @property
    def author(self):
        return self.original_context.author if self.original_context else None

    @property
    def date(self):
        return self.original_context.date if self.original_context else None

    @property
    def origin(self):
        return self.original_context.origin if self.original_context else None

    @property
    def meta_info(self):
        return self.original_context.meta_info if self.original_context else None

    def __str__(self):
        claim_str = f'Claim: "{self.text}"'
        if author := self.author:
            claim_str += f"\nAuthor: {author}"
        if date := self.date:
            claim_str += f"\nDate: {date.strftime('%B %d, %Y')}"
        if origin := self.origin:
            claim_str += f"\nOrigin: {origin}"
        if meta_info := self.meta_info:
            claim_str += f"\nMeta Info: {meta_info}"
        return claim_str
