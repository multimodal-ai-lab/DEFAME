import numpy as np
from sty import fg

"""Colors and formatting for console text"""


def gray(text: str):
    return fg.grey + text + fg.rs


def light_blue(text: str):
    return fg.li_blue + text + fg.rs


def green(text: str):
    return fg.green + text + fg.rs


def red(text: str):
    return fg.red + text + fg.rs


def magenta(text: str):
    return fg.magenta + text + fg.rs


def cyan(text: str):
    return fg.cyan + text + fg.rs


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


def sec2hhmmss(s):
    m = s // 60
    h = m // 60
    return "%d:%02d:%02d h" % (h, m % 60, s % 60)
