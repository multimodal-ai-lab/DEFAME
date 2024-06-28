from dataclasses import dataclass, field
from datetime import datetime
from PIL.Image import Image 

@dataclass
class Claim:
    text: str
    author: str = None
    date: datetime = None
    origin: str = None  # URL or similar
    images: list[Image] = field(default_factory=list)

    def __str__(self):
        claim_str = f'Text: "{self.text}"'
        if self.date:
            claim_str += f"\nClaim date: {self.date.strftime('%B %d, %Y')}"
        if self.author:
            claim_str += f"\nClaim author: {self.author}"
        if self.origin:
            claim_str += f"\nClaim origin: {self.origin}"
        return claim_str
