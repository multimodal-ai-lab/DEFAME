"""Shared utility functions."""

import json
import os
import shutil
import string
from typing import Any
from tqdm import tqdm

from src.utils.console import green, red
from src.utils.parsing import strip_string


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


def maybe_print_error(
        message: str | Exception | None,
        additional_info: str = '',
        verbose: bool = False
) -> None:
    """Prints the error message with additional info if flag is True."""
    if not strip_string(str(message)):
        return

    error = type(message).__name__ if isinstance(message, Exception) else 'ERROR'
    message = f'{error}: {str(message)}'
    message += f'\n{additional_info}' if verbose else ''
    clear_line()
    print(red(message))


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
