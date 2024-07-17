from dataclasses import dataclass

from src.common.content import Content


@dataclass
class Claim:
    text: str  # Atomic, self-contained verbalization of the claim
    original_context: Content  # The input the claim was extracted from

    def __str__(self):
        claim_str = f'Text: "{self.text}"'
        if author := self.original_context.author:
            claim_str += f"\nClaim author: {author}"
        if date := self.original_context.date:
            claim_str += f"\nClaim date: {date.strftime('%B %d, %Y')}"
        if origin := self.original_context.origin:
            claim_str += f"\nClaim origin: {origin}"
        return claim_str

    def has_image(self) -> bool:
        return "<image_" in self.text

    def has_audio(self) -> bool:
        return "<audio_" in self.text

    def has_video(self) -> bool:
        return "<video_" in self.text
