"""Shared utility functions."""

import json
import os
import shutil
import string
from typing import Any
from tqdm import tqdm
from pathlib import Path
import yaml

from infact.utils.console import green, red
from infact.utils.parsing import strip_string


def stop_all_execution(stop_flag: bool) -> None:
    """Immediately stops all execution."""
    if stop_flag:
        print_info('Stopping execution...')
        os._exit(1)


def to_readable_json(json_obj: dict[Any, Any], sort_keys: bool = False) -> str:
    """Converts a json object to a readable string."""
    return f'```json\n{json.dumps(json_obj, indent=2, sort_keys=sort_keys)}\n```'


def recursive_to_saveable(value: Any) -> Any:
    """Converts a value to a saveable value."""
    if isinstance(value, dict):
        return {k: recursive_to_saveable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [recursive_to_saveable(v) for v in value]
    else:
        return str(value)


def open_file_wrapped(filepath: str, **kwargs) -> Any:
    return open(filepath, **kwargs)


def clear_line() -> None:
    """Clears the current line."""
    print(' ' * shutil.get_terminal_size().columns, end='\r')


def print_info(message: str, add_punctuation: bool = True) -> None:
    """Prints the message with an INFO: preamble and colored green."""
    if not message:
        return

    if add_punctuation:
        message = (
            f'{message}.' if message[-1] not in string.punctuation else message
        )
    clear_line()
    print(green(f'INFO: {message}'))


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
