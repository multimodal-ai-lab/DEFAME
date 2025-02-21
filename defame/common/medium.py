import base64
import re
import sqlite3
import warnings
from abc import ABC
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL.Image import Image as PillowImage, open as pillow_open

from config.globals import temp_dir
from defame.utils.parsing import MEDIA_REF_REGEX, MEDIA_SPECIFIER_REGEX
from .logger import logger

warnings.simplefilter("ignore")  # hide all warnings


class Medium(ABC):
    """Superclass of all images, videos, and audios."""
    path_to_file: Path
    data_type: str

    def __init__(self, path_to_file: str | Path):
        self.path_to_file = Path(path_to_file)
        self.id: int = media_registry.add(self)  # automatically add any (unknown) media

    @property
    def reference(self):
        return f"<{self.data_type}:{self.id}>"

    def _load(self) -> None:
        """Loads the medium from the disk using the path."""
        raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other, Medium) and self.path_to_file == other.path_to_file

    def __hash__(self):
        return self.path_to_file.__hash__()


class Image(Medium):
    data_type = "image"
    image: PillowImage

    def __init__(self, path_to_file: str | Path = None, pillow_image: PillowImage = None):
        assert path_to_file is not None or pillow_image is not None

        if pillow_image is not None:
            pillow_image = self._ensure_rgb_mode(pillow_image)
            # Save the image in a temporary folder
            path_to_file = Path(temp_dir) / "media" / (datetime.now().strftime("%Y-%m-%d_%H-%M-%s-%f") + ".jpg")
            path_to_file.parent.mkdir(parents=True, exist_ok=True)
            pillow_image.save(path_to_file)

        super().__init__(path_to_file)

        if pillow_image is not None:
            self.image = pillow_image
        else:
            # Pillow opens images lazily, so actual image read only happens when accessing the image
            pillow_image = pillow_open(self.path_to_file)
            pillow_image = self._ensure_rgb_mode(pillow_image)
            self.image = pillow_image

    def _ensure_rgb_mode(self, pillow_image: PillowImage) -> PillowImage:
        """Turns any kind of image (incl. PNGs) into RGB mode which is JPEG-saveable."""
        if pillow_image.mode != "RGB":
            return pillow_image.convert('RGB')
        else:
            return pillow_image

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

    def __hash__(self):
        return hash(self.image.tobytes())

    def __eq__(self, other):
        return isinstance(other, Image) and np.array_equal(np.array(self.image), np.array(other.image))


class Video(Medium):
    data_type = "video"


class Audio(Medium):
    data_type = "audio"


class MultimediaSnippet:
    """Piece of data holding text which, optionally, contains references
    to media, i.e., images, videos, and audios. Media objects are referenced in the text
    in-line, for example, `<image:0>` for image with ID 0, or `<video:2>`
    for video with ID 2. Unresolvable references will trigger a warning."""
    data: str

    def __init__(self, data: str | list[str | Medium]):
        assert data is not None
        self.data = data if isinstance(data, str) else interleaved_to_string(data)
        if not media_registry.validate(self.data):
            logger.warning("There are unresolvable media references.")

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
        return media_registry.get_media_from_text(self.data, medium_type="image")

    @property
    def videos(self) -> list[Video]:
        return media_registry.get_media_from_text(self.data, medium_type="video")

    @property
    def audios(self) -> list[Audio]:
        return media_registry.get_media_from_text(self.data, medium_type="audio")

    def __str__(self):
        return f"{self.data}"

    def __repr__(self):
        return f"MultimediaSnippet({self.__str__()})"

    def to_interleaved(self) -> list[str | Medium]:
        """Returns a list of interleaved string and media objects representing
        this multimedia snippet. I.e., all the media references are replaced by
        the actual medium object."""
        split = re.split(MEDIA_REF_REGEX, self.data)
        # Replace each reference with its actual medium object
        for i in range(len(split)):
            substr = split[i]
            if is_medium_ref(substr):
                medium = media_registry.get(substr)
                if medium is not None:
                    split[i] = medium
        return split


class MediaRegistry:
    """Keeps track of the paths of all referenced media (images, videos, and audios).
    Also holds a cache of already loaded media files for efficiency."""
    db_location = Path(temp_dir) / "media_registry.db"
    file_name = "media.csv"
    csv_headers = ["medium_type", "id", "path_to_file"]

    def __init__(self):
        # Initialize folder, DB, and cache
        if not self.db_location.parent.exists():
            self.db_location.parent.mkdir(exist_ok=True, parents=True)
        is_new = not self.db_location.exists()
        self.conn = sqlite3.connect(self.db_location, timeout=10, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.cur = self.conn.cursor()
        if is_new:
            self._init_db()
        self.cache: dict[tuple, Medium] = dict()

    def _init_db(self):
        """Initializes a clean, new DB."""
        for medium_type in ["image", "video", "audio"]:
            stmt = f"""
                CREATE TABLE {medium_type}(id INTEGER PRIMARY KEY, path TEXT);
            """
            self.cur.execute(stmt)
            stmt = f"""
                CREATE UNIQUE INDEX {medium_type}_path_idx ON {medium_type}(path);
            """
            self.cur.execute(stmt)
        self.conn.commit()

    def get(self, reference: str) -> Optional[Medium]:
        """Gets the referenced media object by loading it from the cache or,
        if not in the cache, from the disk."""
        medium_type, medium_id = _parse_ref(reference)

        # Read from cache
        medium = self._get_cached(medium_type, medium_id)

        if medium is None:
            # Load from disk
            medium = self._load(medium_type, medium_id)

        return medium

    def add(self, medium: Medium) -> int:
        """Adds a new medium to the registry, if not yet registered. In any case,
        returns the corresponding medium ID."""
        if not self.contains(medium.data_type, medium.path_to_file):
            medium_id = self._insert_into_registry(medium.path_to_file, medium.data_type)
            self._add_to_cache(medium, medium_id)
        else:  # Just return the reference
            medium_id = self._get_id_by_path(medium.data_type, medium.path_to_file)
        return medium_id

    def validate(self, text: str) -> bool:
        """Verifies that each medium reference can be resolved to a registered medium."""
        medium_refs = get_medium_refs(text)
        for ref in medium_refs:
            if self.get(ref) is None:
                return False
        return True

    def get_media_from_text(self, text: str, medium_type: str = None) -> list:
        """Returns the list of all loaded media referenced in the text."""
        medium_refs = get_unique_ordered_medium_refs(text)
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

    def _get_id_by_path(self, medium_type: str, path_to_medium: Path) -> Optional[int]:
        stmt = f"""
            SELECT id
            FROM {medium_type}
            WHERE path = ?;
        """
        response = self.cur.execute(stmt, (_normalize_path(path_to_medium),))
        result = response.fetchone()
        if result is not None:
            return result[0]
        else:
            return None

    def _get_path_by_id(self, medium_type: str, medium_id: int) -> Optional[Path]:
        stmt = f"""
            SELECT path
            FROM {medium_type}
            WHERE id = ?;
        """
        response = self.cur.execute(stmt, (medium_id,))
        result = response.fetchone()
        if result is not None:
            return Path(result[0])
        else:
            return None

    def _insert_into_registry(self, path_to_medium: Path, medium_type: str) -> int:
        """Adds the new medium directly to the database and returns its assigned ID."""
        stmt = f"""
            INSERT INTO {medium_type}(path)
            VALUES (?);
        """
        self.cur.execute(stmt, (_normalize_path(path_to_medium),))
        self.conn.commit()
        return self._get_id_by_path(medium_type, path_to_medium)

    def contains(self, medium_type: str, path_to_medium: Path | str) -> bool:
        return self._get_id_by_path(medium_type, path_to_medium) is not None

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


media_registry = MediaRegistry()


def get_medium_refs(text: str) -> list[str]:
    """Extracts all media references from the text."""
    pattern = re.compile(MEDIA_REF_REGEX, re.DOTALL)
    matches = pattern.findall(text)
    return matches


def get_unique_ordered_medium_refs(text: str) -> list[str]:
    """Extracts all media references from the text, preserving their order and removing duplicates."""
    pattern = re.compile(MEDIA_REF_REGEX, re.DOTALL)
    matches = pattern.findall(text)
    seen = set()
    unique_refs = []
    for match in matches:
        if match not in seen:
            unique_refs.append(match)
            seen.add(match)

    return unique_refs


def parse_media_ref(reference: str) -> Optional[tuple[str, int]]:
    pattern = re.compile(MEDIA_SPECIFIER_REGEX, re.DOTALL)
    result = pattern.findall(reference)
    if len(result) > 0:
        match = result[0]
        return match[0], int(match[1])
    else:
        return None


def is_medium_ref(string: str) -> bool:
    """Returns True iff the string represents a medium reference."""
    pattern = re.compile(MEDIA_REF_REGEX, re.DOTALL)
    return pattern.fullmatch(string) is not None


def interleaved_to_string(interleaved: list[str | Medium]) -> str:
    """Takes a list of strings and media and turns them into a single string
    where each medium is replaced by its reference."""
    substrings = []
    for item in interleaved:
        if isinstance(item, Medium):
            substrings.append(item.reference)
        else:
            substrings.append(item)
    return " ".join(substrings)
