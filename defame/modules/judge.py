import dataclasses
import json
import re
from typing import Optional, Any, Dict, Tuple

from defame.common import Report, logger, Model, Prompt, Label
from defame.common.label import DEFAULT_LABEL_DEFINITIONS
from defame.prompts.prompts import JudgePrompt, JudgeNaively, JudgeMinimal


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_json_block(s: str) -> Optional[str]:
    m = _JSON_BLOCK_RE.search(s)
    return m.group(0) if m else None


def _normalize_verdict_to_label(
    text: Optional[str],
    allowed: Optional[set[Label]] = None
) -> Optional[Label]:
    """
    Map a free-form verdict string to a Label enum, respecting `allowed` if given.
    """
    if not text:
        return None

    t = str(text).strip().lower()

    # common aliases / cleanup
    replacements = {
        "supports": "supported",
        "true": "supported",
        "false": "refuted",
        "not enough info": "not enough information",
        "n/a": "not enough information",
        "unknown": "not enough information",
        "nei": "not enough information",
        "not_enough_information": "not enough information",
        "can't determine": "not enough information",
        "cannot determine": "not enough information",
        "undetermined": "not enough information",
        "insufficient": "not enough information",
        "uncertain": "not enough information",
        "refuse": "refused_to_answer",
        "refused": "refused_to_answer",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)

    # detect by keywords
    candidate: Optional[Label] = None
    if "refused_to_answer" in t:
        candidate = Label.REFUSED_TO_ANSWER
    elif "refut" in t or t in {"fake", "manipulated"}:
        candidate = Label.REFUTED
    elif "supported" in t or t in {"orig", "original", "real", "authentic", "supports"}:
        candidate = Label.SUPPORTED
    elif "not enough information" in t:
        # Prefer NEI if available, else fall back in binary setups
        candidate = Label.NEI if hasattr(Label, "NEI") else Label.REFUTED

    # Allow direct enum names too
    direct_map = {
        "supported": Label.SUPPORTED,
        "refuted": Label.REFUTED,
        "not enough information": Label.NEI if hasattr(Label, "NEI") else Label.REFUTED,
        "refused_to_answer": Label.REFUSED_TO_ANSWER,
    }
    if candidate is None and t in direct_map:
        candidate = direct_map[t]

    # Respect allowed labels if provided
    if candidate is not None and allowed:
        if candidate in allowed:
            return candidate
        # If NEI not allowed, map to REFUTED in binary setups
        if candidate == Label.NEI and Label.REFUTED in allowed:
            return Label.REFUTED
        if candidate == Label.REFUSED_TO_ANSWER and Label.REFUSED_TO_ANSWER in allowed:
            return Label.REFUSED_TO_ANSWER
        # As a last resort, choose a deterministic allowed label
        # (prefer REFUTED if present, else SUPPORTED, else any)
        for pref in (Label.REFUTED, Label.SUPPORTED):
            if pref in allowed:
                return pref
        return next(iter(allowed)) if allowed else candidate

    return candidate


def _heuristic_from_text(
    text: str,
    allowed: Optional[set[Label]] = None
) -> Tuple[Optional[Label], str]:
    """
    Fallback when we don't get parseable JSON. Try to infer the verdict by keywords.
    """
    label = _normalize_verdict_to_label(text, allowed=allowed)
    justification = text.strip()
    return label, justification


class Judge:
    """Determines the truthfulness of a claim given a collection of evidence."""

    def __init__(self,
                 llm: Model,
                 classes: list[Label],
                 class_definitions: Optional[dict[Label, str]] = None,
                 extra_rules: str = None):
        self.llm = llm
        self.classes = set(classes)

        # Ensure class_definitions exists and contains NEI definition if that label exists in your enum
        if class_definitions is None:
            class_definitions = dict(DEFAULT_LABEL_DEFINITIONS)
        else:
            class_definitions = dict(class_definitions)
        if hasattr(Label, "NEI") and Label.NEI not in class_definitions:
            class_definitions[Label.NEI] = DEFAULT_LABEL_DEFINITIONS.get(Label.NEI, "Not enough information.")
        self.class_definitions = class_definitions

        self.extra_rules = extra_rules
        self.max_retries = 5
        self.latest_reasoning = None

    def judge(self, doc: Report, is_final: bool = True) -> Label:
        classes = self.classes.copy()
        # If not final, allow NEI to keep the pipeline going
        if not is_final and hasattr(Label, "NEI"):
            classes.add(Label.NEI)
        prompt = JudgePrompt(doc, classes, self.class_definitions, self.extra_rules)
        return self._generate_verdict(prompt, allowed=classes)

    def judge_naively(self, doc: Report) -> Label:
        prompt = JudgeNaively(doc.claim, self.classes, self.class_definitions)
        return self._generate_verdict(prompt, allowed=self.classes)

    def judge_minimally(self, doc: Report) -> Label:
        prompt = JudgeMinimal(doc.claim, self.classes, self.class_definitions)
        return self._generate_verdict(prompt, allowed=self.classes)

    def _generate_verdict(self, prompt: Prompt, allowed: Optional[set[Label]] = None) -> Label:
        """
        Call the LLM and robustly parse its output into a Label.
        Supports:
          • dict with keys {"verdict","response"/"justification"}
          • raw JSON string
          • text with an embedded JSON block
          • free-form text (heuristic keyword mapping)
        Never raises due to parsing; falls back to REFUSED_TO_ANSWER as a last resort.
        """
        try:
            raw = self.llm.generate(prompt)
        except Exception as e:
            logger.warning(f"[judge] LLM call failed ({e}); defaulting to REFUSED_TO_ANSWER.")
            self.latest_reasoning = ""
            return Label.REFUSED_TO_ANSWER

        # Fast path: dict already
        if isinstance(raw, dict):
            verdict_val = raw.get("verdict") or raw.get("label")
            justification = raw.get("response") or raw.get("justification") or raw.get("reason") or ""
            label = _normalize_verdict_to_label(verdict_val, allowed=allowed)
            if label is None:
                # Try heuristic over the whole dict as text
                label, _ = _heuristic_from_text(json.dumps(raw, ensure_ascii=False), allowed=allowed)
            if label is None:
                logger.warning("[judge] Unrecognized verdict in dict; defaulting to REFUSED_TO_ANSWER.")
                self.latest_reasoning = justification
                return Label.REFUSED_TO_ANSWER
            self.latest_reasoning = justification
            return label

        # Ensure string
        text = "" if raw is None else str(raw)

        # Try direct JSON parse
        resp = _safe_json_loads(text)
        if resp is None:
            # Try JSON-looking block
            block = _extract_json_block(text)
            if block:
                resp = _safe_json_loads(block)

        if isinstance(resp, dict):
            verdict_val = resp.get("verdict") or resp.get("label")
            justification = resp.get("response") or resp.get("justification") or resp.get("reason") or text
            label = _normalize_verdict_to_label(verdict_val, allowed=allowed)
            if label is None:
                label, _ = _heuristic_from_text(text, allowed=allowed)
            if label is None:
                logger.warning("[judge] Unrecognized verdict after JSON parse; defaulting to REFUSED_TO_ANSWER.")
                self.latest_reasoning = justification
                return Label.REFUSED_TO_ANSWER
            self.latest_reasoning = justification
            return label

        # Heuristic fallback on free text
        label, justification = _heuristic_from_text(text, allowed=allowed)
        if label is None:
            logger.warning("[judge] Could not parse verdict; defaulting to REFUSED_TO_ANSWER.")
            self.latest_reasoning = justification
            return Label.REFUSED_TO_ANSWER

        self.latest_reasoning = justification
        return label

    def get_latest_reasoning(self) -> str:
        return self.latest_reasoning if self.latest_reasoning is not None else ""
