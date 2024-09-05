from dataclasses import dataclass

from PIL.Image import Image as PillowImage, open
from pathlib import Path
import base64
from io import BytesIO

from infact.common.media_registry import media_registry


class Medium:
    """Superclass of all images, videos, and audios."""
    path_to_file: Path
    data_type: str

    def __init__(self, path_to_file: str | Path):
        self.path_to_file = Path(path_to_file)

    def _load(self) -> None:
        """Loads the medium from the disk using the path."""
        raise NotImplementedError


class Image(Medium):
    data_type = "image"
    image: PillowImage

    def __init__(self, path_to_file: str | Path):
        super().__init__(path_to_file)
        # Pillow opens images lazily, so actual image read happens when accessing the image
        self.image = open(self.path_to_file)

    def ensure_loaded(self) -> None:
        if self.image is None:
            self._load()

    def get_base64_encoded(self) -> str:
        buffered = BytesIO()
        self.image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


class Video(Medium):
    data_type = "video"


class Audio(Medium):
    data_type = "audio"


@dataclass
class MultimediaSnippet:
    """Piece of data holding text which, optionally, refers to media, i.e.,
    images, videos, and audios. Media objects are referenced in the text
    in-line, for example, `<image:0>` for image with ID 0, or `<video:2>`
    for video with ID 2. Unresolvable references will be deleted automatically,
    logging a warning."""

    text: str

    def __post_init__ (self):
        # Verify if all medium references in the text are valid
        if not media_registry.validate(self.text):
            print("Warning: There are unresolvable media references.")

    def has_images(self) -> bool:
        return len(self.images) > 0

    def has_videos(self) -> bool:
        return len(self.videos) > 0

    def has_audios(self) -> bool:
        return len(self.audios) > 0

    @property
    def images(self) -> list[Image]:
        return media_registry.get_media_from_text(self.text, medium_type="image")

    @property
    def videos(self) -> list[Video]:
        return media_registry.get_media_from_text(self.text, medium_type="video")

    @property
    def audios(self) -> list[Audio]:
        return media_registry.get_media_from_text(self.text, medium_type="audio")

    def is_multimodal(self) -> bool:
        return self.has_images() or self.has_videos() or self.has_audios()

    def __str__(self):
        string_representation = f"\"{self.text}\""
        if self.is_multimodal():
            media_info = []
            if self.has_images():
                media_info.append(f"{len(self.images)} Image(s)")
            if self.has_videos():
                media_info.append(f"{len(self.videos)} Video(s)")
            if self.has_audios():
                media_info.append(f"{len(self.audios)} Audio(s)")
            media_info = ", ".join(media_info)
            string_representation += f"\n{media_info}"
        return string_representation
