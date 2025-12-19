import json
import zipfile
from datetime import datetime

from config.globals import data_root_dir
from defame.common import Label, Claim
from defame.common.medium import Image, Video
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Geolocate, Search


class VeriTaS(Benchmark):
    name = "VeriTaS"
    shorthand = "veritas"

    is_multimodal = True

    # Three-class classification for integrity
    class_mapping = {
        "Intact": Label.INTACT,
        "Compromised": Label.COMPROMISED,
        "Unknown": Label.UNKNOWN,
    }

    class_definitions = {
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

    extra_judge_rules = """* **Holistic Integrity Assessment**: The integrity verdict should reflect:
    - High integrity (Intact): Claim is factually accurate AND any media is authentic and properly contextualized
    - Low integrity (Compromised): Claim is factually inaccurate OR media is manipulated/out-of-context
    - Uncertain integrity (Unknown): Insufficient evidence to make a determination
    * **Media Impact**: Even if text is accurate, misused media can compromise integrity.
    * **Scoring Thresholds**:
      - Intact: integrity >= 0.33
      - Unknown: -0.33 < integrity < 0.33
      - Compromised: integrity <= -0.33
    """

    available_actions = [Search, Geolocate]

    # Thresholds for three-class classification
    INTACT_THRESHOLD = 0.33
    COMPROMISED_THRESHOLD = -0.33

    def __init__(self, variant: str = "q1_2024"):
        """
        Initialize VeriTaS benchmark.

        Args:
            variant: The quarter to use, e.g., 'q1_2024', 'q2_2024', 'q3_2024', 'q4_2024'
        """
        # Extract the zip file if needed
        self.variant = variant
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

    def _load_data(self) -> list[dict]:
        """Load claims from the VeriTaS dataset."""
        print(f"[VeriTaS] Opening claims file: {self.file_path}")
        with open(self.file_path, 'r') as f:
            data_raw = json.load(f)

        metadata = data_raw.get("metadata", {})
        claims = data_raw.get("claims", [])

        print(f"[VeriTaS] Loading {metadata.get('total_claims', len(claims))} claims from VeriTaS {self.variant}...")

        data = []
        for claim_entry in claims:
            claim_id = str(claim_entry["id"])
            claim_text = claim_entry["data"]
            verdict = claim_entry.get("verdict", {})

            # IMPORTANT: Register media files BEFORE creating the Claim object
            # This ensures media references are resolvable when Claim validates them
            try:
                claim_text = self._register_media(claim_text, claim_id)
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

            # Three-class label based on integrity score
            integrity = verdict.get("integrity")
            if integrity is None:
                # Fallback if integrity is missing
                label = Label.UNKNOWN
            elif integrity >= self.INTACT_THRESHOLD:
                label = Label.INTACT
            elif integrity <= self.COMPROMISED_THRESHOLD:
                label = Label.COMPROMISED
            else:
                label = Label.UNKNOWN

            # Store additional ground truth info for evaluation
            justification = {
                "veracity": verdict.get("veracity"),
                "context_coverage": verdict.get("context_coverage"),
                "integrity": integrity,
                "media_verdicts": verdict.get("media", [])
            }

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

    def _register_media(self, claim_text: str, claim_id: str) -> str:
        """
        Register media files referenced in the claim text with the media registry.
        MUST be called BEFORE creating the Claim object.

        Args:
            claim_text: The claim text potentially containing <image:ID> or <video:ID> references
            claim_id: The claim ID for logging purposes

        Returns:
            The claim text with media properly registered
        """
        import re

        # Find all media references in the text
        media_pattern = r'<(image|video):(\d+)>'
        matches = re.findall(media_pattern, claim_text)


        registered_refs = []
        for media_type, media_id in matches:
            # Construct path to media file
            extension = 'jpg' if media_type == 'image' else 'mp4'
            media_path = self.data_dir / "media" / media_type / f"{media_id}.{extension}"

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
