"""Shared utility functions."""

import shutil
from typing import Any
from tqdm import tqdm
from pathlib import Path
import yaml


def my_hook(pbar: tqdm):
    """Wraps tqdm progress bar for urlretrieve()."""

    def update_to(n_blocks=1, block_size=1, total_size=None):
        """
        n_blocks  : int, optional
            Number of blocks transferred so far [default: 1].
        block_size  : int, optional
            Size of each block (in tqdm units) [default: 1].
        total_size  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if total_size is not None:
            pbar.total = total_size
        pbar.update(n_blocks * block_size - pbar.n)

    return update_to


def load_experiment_parameters(from_dir: str | Path):
    config_path = Path(from_dir) / "config.yaml"
    return load_config(config_path)


def load_config(from_path: str | Path):
    with open(from_path, "r") as f:
        return yaml.safe_load(f)


def flatten_dict(to_flatten: dict[str, Any]) -> dict[str, Any]:
    """Flattens a nested dictionary which has string keys. Renames the keys using
    the scheme "<outer_key>/<inner_key>/..."."""
    flat_dict = {}
    for outer_key, outer_value in to_flatten.items():
        if isinstance(outer_value, dict):
            flat_dict_inner = flatten_dict(outer_value)
            for inner_key, inner_value in flat_dict_inner.items():
                flat_dict[f"{outer_key}/{inner_key}"] = inner_value
        else:
            flat_dict[outer_key] = outer_value
    return flat_dict


def unroll_dict(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Inverse function of flatten_dict()."""
    unrolled_dict = {}
    for key, value in flat_dict.items():
        key_parts = key.split("/")
        tmp_dict = unrolled_dict
        for i, key_part in enumerate(key_parts):
            if i == len(key_parts) - 1:  # Deepest dict layer reached
                tmp_dict[key_part] = value
            else:  # Go down one nested layer
                if key_part in tmp_dict:  # Use existing dict
                    tmp_dict = tmp_dict[key_part]
                else:  # Create new dict
                    tmp_dict[key_part] = dict()
                    tmp_dict = tmp_dict[key_part]
    return unrolled_dict


def get_yaml_files(path: str | Path) -> list[Path]:
    """Traverses the directory tree starting at path and returns a list of all
    YAML filepaths contained in it."""
    matches = Path(path).glob("**/*.yaml")
    return [match for match in matches]
