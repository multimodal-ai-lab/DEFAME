from enum import Enum


class Label(Enum):
    SUPPORTED = "supported"
    NEI = "not enough information"
    REFUTED = "refuted"
    CONFLICTING = "conflicting evidence"
    CHERRY_PICKING = "cherry-picking"
    REFUSED_TO_ANSWER = "error: refused to answer"
    OUT_OF_CONTEXT = "out of context"
    MISCAPTIONED = "miscaptioned"
    INTACT = "intact"
    COMPROMISED = "compromised"
    UNKNOWN = "unknown"

    # 7-class integrity labels (with uncertainty levels)
    INTACT_CERTAIN = "intact (certain)"
    INTACT_RATHER_CERTAIN = "intact (rather certain)"
    INTACT_RATHER_UNCERTAIN = "intact (rather uncertain)"
    COMPROMISED_RATHER_UNCERTAIN = "compromised (rather uncertain)"
    COMPROMISED_RATHER_CERTAIN = "compromised (rather certain)"
    COMPROMISED_CERTAIN = "compromised (certain)"


DEFAULT_LABEL_DEFINITIONS = {
    Label.SUPPORTED: "The knowledge from the fact-check supports or at least strongly implies the Claim. "
                     "Mere plausibility is not enough for this decision.",
    Label.NEI: "The fact-check does not contain sufficient information to come to a conclusion. For example, "
               "there is substantial lack of evidence. In this case, state which information exactly "
               "is missing. In particular, if no RESULTS or sources are available, pick this decision.",
    Label.REFUTED: "The knowledge from the fact-check clearly refutes the Claim. The mere absence or lack of "
                   "supporting evidence is not enough reason for being refuted (argument from ignorance).",
    Label.CONFLICTING: "The knowledge from the fact-check contains conflicting evidence from multiple "
                       "RELIABLE sources. Even trying to resolve the conflicting sources through additional "
                       "investigation was not successful.",
    Label.OUT_OF_CONTEXT: "The image is used out of context. This means that while the caption may be factually"
                          "correct, the image does not relate to the caption or is used in a misleading way to "
                          "convey a false narrative.",
    Label.MISCAPTIONED: "The claim has a true image, but the caption does not accurately describe the image, "
                        "providing incorrect information.",
    Label.INTACT: "The claim has intact integrity. The claim is factually accurate, and any media is authentic "
                  "and properly contextualized.",
    Label.COMPROMISED: "The claim has compromised integrity. The claim is factually inaccurate, misleading, or "
                       "contains manipulated/out-of-context media.",
    Label.UNKNOWN: "The integrity of the claim is unknown or uncertain. There is insufficient evidence to "
                   "determine whether the claim is intact or compromised.",

    # 7-class integrity label definitions
    Label.INTACT_CERTAIN: "The claim is factually accurate with strong, unequivocal evidence. "
                          "Any associated media is authentic and properly contextualized.",
    Label.INTACT_RATHER_CERTAIN: "The claim appears factually accurate with strong but not fully definitive evidence. "
                                 "Media appears authentic and properly contextualized.",
    Label.INTACT_RATHER_UNCERTAIN: "The claim weakly appears factually accurate based on limited evidence. "
                                   "There is some indication of integrity but not enough for confidence.",
    Label.COMPROMISED_RATHER_UNCERTAIN: "The claim weakly appears inaccurate or misleading based on limited evidence. "
                                        "There is some indication of compromised integrity but not enough for confidence.",
    Label.COMPROMISED_RATHER_CERTAIN: "The claim appears inaccurate or misleading with strong but not fully definitive evidence. "
                                      "Media appears manipulated or used out of context.",
    Label.COMPROMISED_CERTAIN: "The claim is factually inaccurate, misleading, or contains manipulated/miscontextualized "
                               "media with strong, unequivocal evidence.",
}


# Mapping from 7-class labels to coarsened 3-class labels.
# Only "certain" and "rather certain" levels map to their respective bin;
# everything else (rather uncertain + unknown) maps to Unknown.
COARSEN_7_TO_3 = {
    Label.INTACT_CERTAIN: Label.INTACT,
    Label.INTACT_RATHER_CERTAIN: Label.INTACT,
    Label.INTACT_RATHER_UNCERTAIN: Label.UNKNOWN,
    Label.UNKNOWN: Label.UNKNOWN,
    Label.COMPROMISED_RATHER_UNCERTAIN: Label.UNKNOWN,
    Label.COMPROMISED_RATHER_CERTAIN: Label.COMPROMISED,
    Label.COMPROMISED_CERTAIN: Label.COMPROMISED,
}


# Labels indicating insufficient evidence/confidence — used by procedure
# variants to decide whether to continue iterating.
UNCERTAIN_LABELS = frozenset({
    Label.NEI,
    Label.UNKNOWN,
    Label.INTACT_RATHER_UNCERTAIN,
    Label.COMPROMISED_RATHER_UNCERTAIN,
})


def coarsen_label(label: Label) -> Label:
    """Coarsen a 7-class label to its 3-class equivalent.

    If the label is already 3-class it is returned unchanged.
    """
    return COARSEN_7_TO_3.get(label, label)
