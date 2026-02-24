import json
import zipfile
from datetime import datetime
from pathlib import Path

from config.globals import data_root_dir
from defame.common import Label, Claim
from defame.common.medium import Image, Video
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Geolocate, Search


# ---- 3-class configuration ----

CLASS_MAPPING_3 = {
    "Intact": Label.INTACT,
    "Compromised": Label.COMPROMISED,
    "Unknown": Label.UNKNOWN,
}

CLASS_DEFINITIONS_3 = {
    Label.INTACT:
        "The claim has intact integrity (score >= 0.33). The claim is factually accurate, "
        "and any media is authentic and properly contextualized.",
    Label.COMPROMISED:
        "The claim has compromised integrity (score <= -0.33). The claim is factually inaccurate, "
        "misleading, or contains manipulated/out-of-context media.",
    Label.UNKNOWN:
        "The integrity of the claim is unknown or uncertain (-0.33 < score < 0.33). "
        "There is insufficient evidence to determine whether the claim is intact or compromised."
}

EXTRA_JUDGE_RULES_3 = """* **Holistic Integrity Assessment**: The integrity verdict should reflect:
    - High integrity (Intact): Claim is factually accurate AND any media is authentic and properly contextualized
    - Low integrity (Compromised): Claim is factually inaccurate OR media is manipulated/out-of-context
    - Uncertain integrity (Unknown): Insufficient evidence to make a determination
    * **Media Impact**: Even if text is accurate, misused media can compromise integrity.
    * **Scoring Thresholds**:
      - Intact: integrity >= 0.33
      - Unknown: -0.33 < integrity < 0.33
      - Compromised: integrity <= -0.33
    """

# 3-class thresholds
THRESHOLDS_3 = {
    "intact": 0.33,         # score >= 0.33
    "compromised": -0.33,   # score <= -0.33
}


# ---- 7-class configuration ----

CLASS_MAPPING_7 = {
    "Intact (certain)": Label.INTACT_CERTAIN,
    "Intact (rather certain)": Label.INTACT_RATHER_CERTAIN,
    "Intact (rather uncertain)": Label.INTACT_RATHER_UNCERTAIN,
    "Unknown": Label.UNKNOWN,
    "Compromised (rather uncertain)": Label.COMPROMISED_RATHER_UNCERTAIN,
    "Compromised (rather certain)": Label.COMPROMISED_RATHER_CERTAIN,
    "Compromised (certain)": Label.COMPROMISED_CERTAIN,
}

CLASS_DEFINITIONS_7 = {
    Label.INTACT_CERTAIN:
        "The claim is factually accurate with strong, unequivocal evidence. "
        "Any associated media is authentic and properly contextualized.",
    Label.INTACT_RATHER_CERTAIN:
        "The claim appears factually accurate with strong but not fully definitive evidence. "
        "Media appears authentic and properly contextualized.",
    Label.INTACT_RATHER_UNCERTAIN:
        "The claim weakly appears factually accurate based on limited evidence. "
        "There is some indication of integrity but not enough for confidence.",
    Label.UNKNOWN:
        "There is insufficient evidence to determine the claim's accuracy or integrity.",
    Label.COMPROMISED_RATHER_UNCERTAIN:
        "The claim weakly appears inaccurate or misleading based on limited evidence. "
        "There is some indication of compromised integrity but not enough for confidence.",
    Label.COMPROMISED_RATHER_CERTAIN:
        "The claim appears inaccurate or misleading with strong but not fully definitive evidence. "
        "Media appears manipulated or used out of context.",
    Label.COMPROMISED_CERTAIN:
        "The claim is factually inaccurate, misleading, or contains manipulated/miscontextualized "
        "media with strong, unequivocal evidence.",
}

EXTRA_JUDGE_RULES_7 = """* **Holistic Integrity Assessment with Uncertainty**: The integrity verdict should reflect
    both the direction (intact vs compromised) and your confidence level (certain, rather certain, rather uncertain).
    - **Intact (certain)**: Claim is factually accurate AND any media is authentic with strong, unequivocal evidence
    - **Intact (rather certain)**: Claim appears accurate with strong but not fully definitive evidence
    - **Intact (rather uncertain)**: Claim weakly appears accurate based on limited evidence
    - **Unknown**: Insufficient evidence to determine integrity in either direction
    - **Compromised (rather uncertain)**: Claim weakly appears inaccurate based on limited evidence
    - **Compromised (rather certain)**: Claim appears inaccurate with strong but not fully definitive evidence
    - **Compromised (certain)**: Claim is clearly inaccurate, misleading, or contains manipulated media
    * **Media Impact**: Even if text is accurate, misused media can compromise integrity.
    * **Confidence Calibration**: Choose the uncertainty level that best reflects the strength of available evidence.
      Only use "certain" when evidence is overwhelming and unambiguous.
    """

# 7-class thresholds (matching discretize_7_bins from Veritas)
THRESHOLDS_7 = [
    (-5 / 6, Label.COMPROMISED_CERTAIN),
    (-3 / 6, Label.COMPROMISED_RATHER_CERTAIN),
    (-1 / 6, Label.COMPROMISED_RATHER_UNCERTAIN),
    (1 / 6, Label.UNKNOWN),
    (3 / 6, Label.INTACT_RATHER_UNCERTAIN),
    (5 / 6, Label.INTACT_RATHER_CERTAIN),
    (float('inf'), Label.INTACT_CERTAIN),
]


def _classify_integrity_7(score: float) -> Label:
    """Classify an integrity score into a 7-class label using threshold bins."""
    for threshold, label in THRESHOLDS_7:
        if score < threshold:
            return label
    return Label.INTACT_CERTAIN


class VeriTaS(Benchmark):
    name = "VeriTaS"
    shorthand = "veritas"

    is_multimodal = True

    # Defaults (overridden in __init__ based on label_scheme)
    class_mapping = CLASS_MAPPING_3
    class_definitions = CLASS_DEFINITIONS_3

    extra_prepare_rules = """**Multimodal Integrity Assessment**: Evaluate the overall integrity of the claim by considering:
    - **Veracity**: Is the textual claim factually accurate?
    - **Media Authenticity**: Are any images/videos genuine or manipulated?
    - **Media Contextualization**: Is the media used in the proper context or taken out of context?
    The overall integrity combines all these factors."""

    extra_plan_rules = """* **Comprehensive Verification**: For each claim, verify:
    1. The factual accuracy of the text claim (use web search)
    2. The authenticity of any referenced media (use reverse image search)
    3. The proper contextualization of media (verify the original context)
    * **Multimodal Claims**: Pay special attention to claims with media - verify both text and visual content.
    """

    extra_judge_rules = EXTRA_JUDGE_RULES_3

    available_actions = [Search, Geolocate]

    # Thresholds for three-class classification (legacy)
    INTACT_THRESHOLD = 0.33
    COMPROMISED_THRESHOLD = -0.33

    def __init__(self, variant: str = "q1_2024", data_path: str = None, label_scheme: int = 3):
        """
        Initialize VeriTaS benchmark.

        Args:
            variant: The quarter to use, e.g., 'q1_2024', 'q2_2024', 'q3_2024', 'q4_2024',
                     or 'longitudinal' for the new longitudinal format
            data_path: Optional explicit path to a claims.json file or directory containing it.
                       If provided, this overrides the default path resolution.
            label_scheme: Number of classes for the label scheme (3 or 7). Default: 3.
        """
        if label_scheme not in (3, 7):
            raise ValueError(f"Unsupported label_scheme: {label_scheme}. Choose 3 or 7.")

        self.label_scheme = label_scheme

        # Configure class mapping, definitions, and judge rules based on label scheme
        if label_scheme == 7:
            self.class_mapping = CLASS_MAPPING_7
            self.class_definitions = CLASS_DEFINITIONS_7
            self.extra_judge_rules = EXTRA_JUDGE_RULES_7
        else:
            self.class_mapping = CLASS_MAPPING_3
            self.class_definitions = CLASS_DEFINITIONS_3
            self.extra_judge_rules = EXTRA_JUDGE_RULES_3

        self.variant = variant
        self._data_format = None  # Will be detected: 'legacy' or 'longitudinal'

        if data_path:
            # Use explicit data path
            data_path = Path(data_path)
            if data_path.is_file():
                self.file_path = data_path
                self.data_dir = data_path.parent
            else:
                self.data_dir = data_path
                self.file_path = data_path / "claims.json"

            if not self.file_path.exists():
                raise ValueError(f"Claims file not found at {self.file_path}")

            # Skip parent __init__ file_path handling since we set it directly
            self.full_name = f"{self.name} ({variant})"
            self.data = self._load_data()
        else:
            # Legacy path resolution with zip extraction
            self.data_dir = data_root_dir / "VeriTaS" / variant
            self._ensure_extracted(variant)
            super().__init__(variant=variant, file_path=f"VeriTaS/{variant}/claims.json")

    def _ensure_extracted(self, variant: str):
        """Extract the quarterly zip file if not already extracted."""
        zip_path = data_root_dir / "VeriTaS" / f"veritas_benchmark_{variant}.zip"
        extract_dir = data_root_dir / "VeriTaS" / variant

        if not extract_dir.exists():
            if not zip_path.exists():
                raise ValueError(
                    f"VeriTaS data for {variant} not found at {zip_path}. "
                    f"Please download the VeriTaS benchmark data."
                )

            print(f"Extracting {variant} data...", end="")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(" done.")

    def _detect_format(self, claim_entry: dict) -> str:
        """Detect the data format based on claim structure."""
        if self._data_format:
            return self._data_format

        # Legacy format uses "data" field with inline media refs like <image:ID>
        # New longitudinal format uses "text" field with separate "media" array
        if "text" in claim_entry and "data" not in claim_entry:
            self._data_format = "longitudinal"
        elif "data" in claim_entry:
            self._data_format = "legacy"
        else:
            raise ValueError(f"Unknown data format: claim has neither 'text' nor 'data' field")

        print(f"[VeriTaS] Detected data format: {self._data_format}")
        return self._data_format

    def _get_integrity_score(self, claim_entry: dict) -> float | None:
        """Extract integrity score from claim entry, handling both formats."""
        if self._data_format == "longitudinal":
            # New format: integrity is a dict with "score" and "decisive_property"
            integrity_obj = claim_entry.get("integrity")
            if integrity_obj is None:
                return None
            if isinstance(integrity_obj, dict):
                return integrity_obj.get("score")
            return integrity_obj
        else:
            # Legacy format: integrity is in verdict dict
            verdict = claim_entry.get("verdict", {})
            return verdict.get("integrity")

    def _get_claim_text(self, claim_entry: dict) -> str:
        """Extract claim text from claim entry, handling both formats."""
        if self._data_format == "longitudinal":
            return claim_entry.get("text", "")
        else:
            return claim_entry.get("data", "")

    def _build_claim_text_with_media(self, claim_entry: dict, claim_id: str) -> str:
        """
        Build claim text with inline media references for the longitudinal format.

        The longitudinal format has media as a separate array, but DEFAME expects
        inline references like <image:ID> in the claim text.
        """
        claim_text = self._get_claim_text(claim_entry)

        if self._data_format == "longitudinal":
            # Build media references from the media array
            media_list = claim_entry.get("media", [])
            media_refs = []

            for media_item in media_list:
                media_type = media_item.get("type")
                media_id = media_item.get("id")

                if media_type and media_id:
                    media_refs.append(f"<{media_type}:{media_id}>")

            # Prepend media references to the claim text
            if media_refs:
                claim_text = " ".join(media_refs) + " " + claim_text

        return claim_text

    def _get_media_path(self, media_type: str, media_id: str, claim_entry: dict = None) -> Path:
        """Get the path to a media file, handling both formats."""
        if self._data_format == "longitudinal":
            # New format: media files in images/ or videos/ subdirectories
            extension = 'jpg' if media_type == 'image' else 'mp4'
            return self.data_dir / f"{media_type}s" / f"{media_id}.{extension}"
        else:
            # Legacy format: media files in media/image/ or media/video/ subdirectories
            extension = 'jpg' if media_type == 'image' else 'mp4'
            return self.data_dir / "media" / media_type / f"{media_id}.{extension}"

    def _get_justification(self, claim_entry: dict) -> dict:
        """Extract justification/ground truth info from claim entry."""
        if self._data_format == "longitudinal":
            # New format: veracity and context_coverage at claim level
            integrity_obj = claim_entry.get("integrity", {})
            integrity_score = integrity_obj.get("score") if isinstance(integrity_obj, dict) else integrity_obj

            # Build media verdicts from the new format
            media_verdicts = []
            for media_item in claim_entry.get("media", []):
                authenticity = media_item.get("authenticity", {})
                contextualization = media_item.get("contextualization", {})
                media_verdicts.append({
                    "media_id": media_item.get("id"),
                    "media_type": media_item.get("type"),
                    "authenticity": authenticity.get("score") if isinstance(authenticity, dict) else authenticity,
                    "contextualization": contextualization.get("score") if isinstance(contextualization, dict) else contextualization,
                })

            veracity_obj = claim_entry.get("veracity")
            context_obj = claim_entry.get("context_coverage")

            return {
                "veracity": veracity_obj.get("score") if isinstance(veracity_obj, dict) else veracity_obj,
                "context_coverage": context_obj.get("score") if isinstance(context_obj, dict) else context_obj,
                "integrity": integrity_score,
                "media_verdicts": media_verdicts
            }
        else:
            # Legacy format
            verdict = claim_entry.get("verdict", {})
            return {
                "veracity": verdict.get("veracity"),
                "context_coverage": verdict.get("context_coverage"),
                "integrity": verdict.get("integrity"),
                "media_verdicts": verdict.get("media", [])
            }

    def _load_data(self) -> list[dict]:
        """Load claims from the VeriTaS dataset."""
        print(f"[VeriTaS] Opening claims file: {self.file_path}")
        with open(self.file_path, 'r') as f:
            data_raw = json.load(f)

        metadata = data_raw.get("metadata", {})
        claims = data_raw.get("claims", [])

        print(f"[VeriTaS] Loading {metadata.get('total_claims', len(claims))} claims from VeriTaS {self.variant}...")

        # Detect format from first claim
        if claims:
            self._detect_format(claims[0])

        data = []
        for claim_entry in claims:
            claim_id = str(claim_entry["id"])

            # Build claim text with inline media references
            claim_text = self._build_claim_text_with_media(claim_entry, claim_id)

            # IMPORTANT: Register media files BEFORE creating the Claim object
            # This ensures media references are resolvable when Claim validates them
            try:
                claim_text = self._register_media(claim_text, claim_id, claim_entry)
            except Exception as e:
                print(f"[VeriTaS] ERROR registering media for claim {claim_id}: {e}")
                raise

            # Create Claim object (will now validate media references successfully)
            date_str = claim_entry.get("date")
            try:
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00')) if date_str else None
            except Exception as e:
                print(f"[VeriTaS] ERROR parsing date '{date_str}' for claim {claim_id}: {e}")
                date = None

            try:
                claim = Claim(
                    data=claim_text,
                    date=date,
                    id=claim_id
                )
            except Exception as e:
                print(f"[VeriTaS] ERROR creating Claim object for claim {claim_id}: {e}")
                print(f"[VeriTaS] Claim text: {claim_text}")
                raise

            # Label based on integrity score (3-class or 7-class)
            integrity = self._get_integrity_score(claim_entry)
            if integrity is None:
                label = Label.UNKNOWN
            elif self.label_scheme == 7:
                label = _classify_integrity_7(integrity)
            elif integrity >= self.INTACT_THRESHOLD:
                label = Label.INTACT
            elif integrity <= self.COMPROMISED_THRESHOLD:
                label = Label.COMPROMISED
            else:
                label = Label.UNKNOWN

            # Store additional ground truth info for evaluation
            justification = self._get_justification(claim_entry)

            data.append({
                "id": claim_id,
                "input": claim,
                "label": label,
                "justification": justification
            })

        print(f"[VeriTaS] Successfully loaded {len(data)} claims")
        print(f"[VeriTaS] Label distribution:")
        label_counts = {}
        for item in data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        for label, count in label_counts.items():
            print(f"[VeriTaS]   {label}: {count}")

        return data

    def _register_media(self, claim_text: str, claim_id: str, claim_entry: dict = None) -> str:
        """
        Register media files referenced in the claim text with the media registry.
        MUST be called BEFORE creating the Claim object.

        Args:
            claim_text: The claim text potentially containing <image:ID> or <video:ID> references
            claim_id: The claim ID for logging purposes
            claim_entry: Optional claim entry dict (used for format-specific path resolution)

        Returns:
            The claim text with media properly registered
        """
        import re

        # Find all media references in the text
        media_pattern = r'<(image|video):(\d+)>'
        matches = re.findall(media_pattern, claim_text)

        registered_refs = []
        for media_type, media_id in matches:
            # Construct path to media file using format-aware method
            media_path = self._get_media_path(media_type, media_id, claim_entry)

            if media_path.exists():
                try:
                    # Register the media with the global media registry
                    # This creates the media object and assigns it a registry ID
                    if media_type == "image":
                        media_obj = Image(media_path)
                    elif media_type == "video":
                        media_obj = Video(media_path)
                    else:
                        print(f"[VeriTaS]   WARNING: Unknown media type '{media_type}'")
                        continue

                    # Store the reference for replacement if needed
                    old_ref = f"<{media_type}:{media_id}>"
                    new_ref = media_obj.reference
                    registered_refs.append((old_ref, new_ref))
                except Exception as e:
                    print(f"[VeriTaS] ERROR registering media {media_path}: {e}")
                    raise
            else:
                print(f"[VeriTaS] WARNING: Media file not found for claim {claim_id}: {media_path}")
                from defame.common import logger
                logger.warning(f"Media file not found for claim {claim_id}: {media_path}")

        # Replace old references with new registry references if different
        for old_ref, new_ref in registered_refs:
            if old_ref != new_ref:
                claim_text = claim_text.replace(old_ref, new_ref)

        return claim_text
