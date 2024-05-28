from enum import Enum


class Label(Enum):
    SUPPORTED = "supported"
    NEI = "not enough information"
    REFUTED = "refuted"
    CONFLICTING = "conflicting evidence"
    REFUSED_TO_ANSWER = "refused"
