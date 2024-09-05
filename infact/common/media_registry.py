import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from infact.common.medium import Medium, Image, Video, Audio
from infact.utils.parsing import parse_media_ref, get_medium_refs
from config.globals import temp_dir


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

    def get_media_from_text(self, text: str, medium_type: str = None) -> list[Medium]:
        """Returns the list of all loaded media referenced in the text."""
        medium_refs = get_medium_refs(text)
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
