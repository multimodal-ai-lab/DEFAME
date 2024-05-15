from enum import Enum


class Label(Enum):
    SUPPORTED = "Supported"
    NEI = "Not enough information"
    REFUTED = "Refuted"
    CONFLICTING = "Conflicting evidence"
