from dataclasses import dataclass
from datetime import datetime

from PIL.Image import Image


@dataclass
class Content:
    """The raw content to be interpreted, decomposed, decontextualized, etc. before
    being checked. Media resources are referred in the text in-line as follows:
    <image_0> for images[0], <video_2> for videos[2], etc."""
    text: str
    images: list[Image] = None
    videos = None
    audios = None

    author: str = None
    date: datetime = None
    origin: str = None  # URL

    interpretation: str = None  # Added during claim extraction

    def is_multimodal(self) -> bool:
        return self.images is not None or self.videos is not None or self.audios is not None

    def __str__(self):
        summary = []
        if self.images:
            summary.append(f"{len(self.images)} Image(s)")
        if self.videos:
            summary.append(f"{len(self.videos)} Video(s)")
        if self.audios:
            summary.append(f"{len(self.audios)} Audio(s)")
        summary = ", ".join(summary)
        return f"Content: {summary}\n\"{self.text}\""
