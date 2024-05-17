from enum import Enum


class Label(Enum):
    SUPPORTED = "Supported"
    NEI = "Not enough information"
    REFUTED = "Refuted"
    CONFLICTING = "Conflicting evidence"
    REFUSED_TO_ANSWER = "Refused"
