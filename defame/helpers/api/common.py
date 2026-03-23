from typing import Optional, Annotated

from ezmm import MultimodalSequence, Image
from pydantic import BaseModel, Field

from defame.common import Label, Claim, Content

MultimediaSequence = Annotated[
    list[tuple[str, str]],
    Field(
        description="Interleaved sequence of text and media.",
        examples=[
            [("text", "This is some natural language text, followed by an image:"),
             ("image", "<base64_encoded_image_string>"),
             ("text", "The image is actually a string, encoded in Base64.")],
        ])
]


class UserSubmission(BaseModel):
    """User-submitted, raw content to be fact-checked."""
    content: MultimediaSequence
    author: Optional[str] = None
    date: Optional[str] = None


### Info wrappers for common objects (for API serialization)


class ClaimInfo(BaseModel):
    claim_id: str
    data: MultimediaSequence
    verdict: Optional[str] = Field(default=None, examples=[Label.SUPPORTED.name,
                                                           Label.REFUTED.name,
                                                           Label.NEI.name,
                                                           Label.CHERRY_PICKING.name])
    justification: Optional[MultimediaSequence] = None


class ContentInfo(BaseModel):
    data: MultimediaSequence
    author: Optional[str] = None
    date: Optional[str] = Field(default=None, examples=["2025-02-21"], description="Format is YYYY-MM-DD")
    topic: Optional[str] = None
    verdict: Optional[str] = Field(default=None, examples=[Label.SUPPORTED.name,
                                                           Label.REFUTED.name,
                                                           Label.NEI.name,
                                                           Label.CHERRY_PICKING.name],
                                   description="The overall veracity, aggregated from all claims.")


def get_claim_info(claim: Claim) -> ClaimInfo:
    return ClaimInfo(
        claim_id=claim.id,
        data=to_sequence(claim),
        verdict=claim.verdict,
        justification=to_sequence(claim.justification) if claim.justification else None
    )


def get_content_info(content: Content) -> ContentInfo:
    return ContentInfo(
        data=to_sequence(content),
        author=content.author,
        date=content.date.strftime('%Y-%m-%d') if content.date else None,
        topic=content.topic,
        verdict=content.verdict
    )


def to_sequence(multimedia_snippet: MultimodalSequence) -> MultimediaSequence:
    interleaved = multimedia_snippet.to_list()
    data = []
    for block in interleaved:
        if isinstance(block, str):
            data.append(("text", block))
        elif isinstance(block, Image):
            data.append((block.kind, block.get_base64_encoded()))
        else:
            raise NotImplementedError("Audio and video not supported yet.")
    return data
