"""Colors and formatting for console text"""

import re
from typing import Optional

import numpy as np
from sty import fg, Style, RgbFg

fg.orange = Style(RgbFg(255, 150, 50))


def gray(text: str):
    return fg.da_grey + text + fg.rs


def light_blue(text: str):
    return fg.li_blue + text + fg.rs


def green(text: str):
    return fg.green + text + fg.rs


def yellow(text: str):
    return fg.li_yellow + text + fg.rs


def red(text: str):
    return fg.red + text + fg.rs


def magenta(text: str):
    return fg.magenta + text + fg.rs


def cyan(text: str):
    return fg.cyan + text + fg.rs


def orange(text: str):
    return fg.orange + text + fg.rs


def bold(text):  # boldface
    return "\033[1m" + text + "\033[0m"


def it(text):  # italic
    return "\033[3m" + text + "\033[0m"


def ul(text):  # underline
    return "\033[4m" + text + "\033[0m"


def num2text(num):
    if num == 0:
        return "0"
    elif np.abs(num) < 1:
        return "%.2f" % num
    elif np.abs(num) < 10 and num % 1 != 0:
        return "%.1f" % num
    elif np.abs(num) < 1000:
        return "%.0f" % num
    elif np.abs(num) < 10000:
        thousands = num / 1000
        return "%.1fK" % thousands
    elif np.abs(num) < 1e6:
        thousands = num / 1000
        return "%.0fK" % thousands
    elif np.abs(num) < 1e7:
        millions = num / 1e6
        return "%.1fM" % millions
    else:
        millions = num / 1e6
        return "%.0fM" % millions


def sec2hhmmss(s: float) -> Optional[str]:
    if s is None:
        return None
    m = s // 60
    h = m // 60
    return "%d:%02d:%02d h" % (h, m % 60, s % 60)


def sec2mmss(s: float) -> Optional[str]:
    if s is None:
        return None
    m = s // 60
    return "%d:%02d min" % (m % 60, s % 60)


def remove_string_formatters(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)
