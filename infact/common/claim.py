from dataclasses import dataclass

from infact.common.content import Content
from infact.common.medium import MultimediaSnippet


@dataclass
class Claim(MultimediaSnippet):
    original_context: Content = None  # The input the claim was extracted from

    def __str__(self):
        claim_str = f'Claim: "{self.text}"'
        if author := self.original_context.author:
            claim_str += f"\nAuthor: {author}"
        if date := self.original_context.date:
            claim_str += f"\nDate: {date.strftime('%B %d, %Y')}"
        if origin := self.original_context.origin:
            claim_str += f"\nOrigin: {origin}"
        return claim_str
