from dataclasses import dataclass

from defame.common.content import Content
from defame.common.medium import MultimediaSnippet


@dataclass
class Claim(MultimediaSnippet):
    original_context: Content  # The input the claim was extracted from

    @property
    def author(self):
        return self.original_context.author

    @property
    def date(self):
        return self.original_context.date

    def __str__(self):
        claim_str = f'Claim: "{self.text}"'
        if author := self.original_context.author:
            claim_str += f"\nAuthor: {author}"
        if date := self.original_context.date:
            claim_str += f"\nDate: {date.strftime('%B %d, %Y')}"
        if origin := self.original_context.origin:
            claim_str += f"\nOrigin: {origin}"
        if meta_info := self.original_context.meta_info:
            claim_str += f"\nMeta Info: {meta_info}"
        return claim_str
