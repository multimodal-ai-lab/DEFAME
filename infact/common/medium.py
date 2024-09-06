import base64
import csv
from abc import ABC
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
from PIL.Image import Image as PillowImage, open as pillow_open

from config.globals import temp_dir
from infact.utils.parsing import get_medium_refs, parse_media_ref


class Medium(ABC):
    """Superclass of all images, videos, and audios."""
    path_to_file: Path
    data_type: str

    def __init__(self, path_to_file: str | Path):
        self.path_to_file = Path(path_to_file)
        self.reference = None
        self.id = None

    def _load(self) -> None:
        """Loads the medium from the disk using the path."""
        raise NotImplementedError

    def __eq__(self, other):
        return self.path_to_file == other.path_to_file

    def __hash__(self):
        return self.path_to_file.__hash__()


class Image(Medium):
    data_type = "image"
    image: PillowImage

    def __init__(self, path_to_file: str | Path = None, pillow_image: PillowImage = None):
        assert path_to_file is not None or pillow_image is not None

        if pillow_image is not None:
            # Save the image in a temporary folder
            path_to_file = Path(temp_dir) / "media" / (datetime.now().strftime("%Y-%m-%d_%H-%M-%s-%f") + ".jpg")
            path_to_file.parent.mkdir(parents=True, exist_ok=True)
            pillow_image.save(path_to_file)

        super().__init__(path_to_file)

        if pillow_image is not None:
            self.image = pillow_image
        else:
            # Pillow opens images lazily, so actual image read only happens when accessing the image
            self.image = pillow_open(self.path_to_file)

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

    def __post_init__(self):
        # Verify if all medium references in the text are valid
        if not media_registry.validate(self.text):
            print("Warning: There are unresolvable media references.")

    def has_images(self) -> bool:
        return len(self.images) > 0

    def has_videos(self) -> bool:
        return len(self.videos) > 0

    def has_audios(self) -> bool:
        return len(self.audios) > 0

    def is_multimodal(self) -> bool:
        return self.has_images() or self.has_videos() or self.has_audios()

    @property
    def images(self) -> list[Image]:
        return media_registry.get_media_from_text(self.text, medium_type="image")

    @property
    def videos(self) -> list[Video]:
        return media_registry.get_media_from_text(self.text, medium_type="video")

    @property
    def audios(self) -> list[Audio]:
        return media_registry.get_media_from_text(self.text, medium_type="audio")

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


class MediaRegistry:
    """Keeps track of the paths of all referenced media (images, videos, and audios).
    Also holds a cache of already loaded media files for efficiency."""
    file_name = "media.csv"
    csv_headers = ["medium_type", "id", "path_to_file"]

    def __init__(self, target_dir: Path | str):
        # Initialize folder and file
        target_dir = Path(target_dir)
        if not target_dir.exists():
            target_dir.mkdir()
        self.path = target_dir / self.file_name
        if not self.path.exists():
            with open(self.path, "w") as f:
                csv.writer(f).writerow(self.csv_headers)

        self.cache = dict()

    def get(self, reference: str) -> Optional[Medium]:
        """Gets the referenced media object by loading it from the cache or,
        if not in the cache, from the disk."""
        medium_type, medium_id = _parse_ref(reference)

        # Read from cache
        medium = self._get_cached(medium_type, medium_id)

        if medium is None:
            # Load from disk
            medium = self._load(medium_type, medium_id)
            if medium is not None:
                self._add_to_cache(medium, medium_id)

        # Add reference to medium
        if medium is not None:
            medium.reference = reference
            medium.id = medium_id

        return medium

    def add(self, medium: Medium) -> str:
        """Adds a new medium to the registry, if not yet registered. In any case,
        returns the corresponding reference."""
        if not self.contains(medium.path_to_file):
            medium_id = self._insert_into_registry(medium.path_to_file, medium.data_type)
            if medium is not None:
                self._add_to_cache(medium, medium_id)
        else:  # Just return the reference
            medium_id = self._get_id_by_path(medium.path_to_file)
            assert medium_id is not None

        return f"<{medium.data_type}:{medium_id}>"

    def validate(self, text: str) -> bool:
        """Verifies that each medium reference can be resolved to a registered medium."""
        medium_refs = get_medium_refs(text)
        for ref in medium_refs:
            if self.get(ref) is None:
                return False
        return True

    def get_media_from_text(self, text: str, medium_type: str = None) -> list:
        """Returns the list of all loaded media referenced in the text."""
        medium_refs = set(get_medium_refs(text))
        media = []
        for ref in medium_refs:
            medium = self.get(ref)
            if medium is not None and (medium_type is None or medium.data_type == medium_type):
                media.append(medium)
        return media

    def _load(self, medium_type: str, medium_id: int) -> Optional[Medium]:
        """Loads the specified medium from the disk."""
        path_to_medium = self._get_path_by_id(medium_type, medium_id)
        if path_to_medium is not None and path_to_medium.exists():
            match medium_type:
                case Image.data_type:
                    return Image(path_to_medium)
                case Video.data_type:
                    return Video(path_to_medium)
                case Audio.data_type:
                    return Audio(path_to_medium)
        return None

    def _get_id_by_path(self, path_to_medium: Path) -> Optional[int]:
        matches = self.media["path_to_file"] == _normalize_path(path_to_medium)
        if not np.any(matches):
            return None
        return self.media[matches]["id"].values[0]

    def _get_path_by_id(self, medium_type: str, medium_id: int) -> Optional[Path]:
        media = self.media
        matches = (media["medium_type"] == medium_type) & (media["id"] == medium_id)
        if not np.any(matches):
            return None
        return Path(media[matches]["path_to_file"].values[0])

    def _insert_into_registry(self, path_to_medium: Path, medium_type: str) -> int:
        """Adds the new medium directly to the media.csv file."""
        new_id = self.get_max_id(medium_type) + 1
        with open(self.path, "a") as f:
            csv.writer(f).writerow([medium_type, new_id, _normalize_path(path_to_medium)])
        return new_id

    def get_max_id(self, medium_type: str) -> int:
        media = self.media
        matches = media["medium_type"] == medium_type
        if not np.any(matches):
            return -1
        else:
            return media[matches]["id"].max()

    def contains(self, path_to_medium: Path | str) -> bool:
        matches = self.media["path_to_file"] == _normalize_path(path_to_medium)
        return np.any(matches)

    def _load_registry(self):
        if not self.path.exists():
            return self._empty_registry()
        else:
            return pd.read_csv(self.path)

    def _empty_registry(self):
        return pd.DataFrame(columns=self.csv_headers)

    @property
    def media(self):
        return self._load_registry()

    def _get_cached(self, medium_type: str, medium_id: int) -> Optional[Medium]:
        """Tries to retrieve media object from the cache. Returns
        None if it is not in the cache."""
        return self.cache.get((medium_type, medium_id))

    def _add_to_cache(self, medium: Medium, medium_id: int) -> None:
        """Adds a media object to the cache."""
        medium_type = medium.data_type
        self.cache[(medium_type, medium_id)] = medium


def _parse_ref(ref: str) -> (str, int):
    result = parse_media_ref(ref)
    if result is None:
        raise ValueError(f"Invalid media reference: {ref}")
    return result


def _normalize_path(path: Path | str) -> str:
    """Converts the path into a POSIX string representation of the absolute path."""
    path = Path(path)
    return path.absolute().as_posix()


media_registry = MediaRegistry(temp_dir)
