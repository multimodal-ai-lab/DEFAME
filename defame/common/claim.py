from typing import Optional

from defame.common.label import Label
from defame.common.content import Content
from defame.common.medium import MultimediaSnippet


class Claim(MultimediaSnippet):
    context: Optional[Content]  # Original content where the claim was extracted from
    scope: Optional[tuple[int, int]]  # TODO: the range in the original context's text that belongs to this claim
    id: Optional[str]

    def __init__(self, *args,
                 identifier: str | int | None = None,
                 context: Content = None,
                 scope: tuple[int, int] = None,
                 **kwargs):
        self.id = str(identifier) if identifier is not None else None
        self.context = context
        self.scope = scope
        self.verdict: Optional[Label] = None
        self.justification: Optional[MultimediaSnippet] = None
        super().__init__(*args, **kwargs)

    @property
    def author(self):
        return self.context.author if self.context else None

    @property
    def date(self):
        return self.context.date if self.context else None

    @property
    def origin(self):
        return self.context.origin if self.context else None

    @property
    def meta_info(self):
        return self.context.meta_info if self.context else None

    def __str__(self):
        claim_str = f'Claim: "{self.data}"'
        if author := self.author:
            claim_str += f"\nAuthor: {author}"
        if date := self.date:
            claim_str += f"\nDate: {date.strftime('%B %d, %Y')}"
        if origin := self.origin:
            claim_str += f"\nOrigin: {origin}"
        if meta_info := self.meta_info:
            claim_str += f"\nMeta info: {meta_info}"
        return claim_str

    def __repr__(self):
        return f"Claim(\"{self.data}\", context={self.context.__repr__()})"

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary.update(dict(  # id=self.id,
            scope=self.scope,
            verdict=self.verdict,
            justification=self.justification
        ))
        return summary
