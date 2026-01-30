# =============================================================================
# Image Evidence via Metadata (LLM + ExifTool) — Robust version with local fallback
#
# PURPOSE (for LLM planners / tool-using agents)
# -----------------------------------------------------------------------------
# • This tool is MANDATORY for any claim that includes (or references) an IMAGE.
# • LLM planners MUST always schedule this tool whenever an image is present in
#   the current claim/evidence bundle.
# • It produces a compact, structured SUMMARY of EXIF/metadata + pixel-level cues
#   to aid a fact-checker. This tool is NOT a classifier and MUST NOT emit a
#   “real/fake” judgment.
#
# WHEN TO CALL
# -----------------------------------------------------------------------------
# • If the current Claim/Evidence bundle includes an image or a resolvable image
#   reference (local path/URL supported by ezmm.Image).
# • If multiple images exist, call the tool ONCE PER IMAGE (one
#   VerifyImageAuthenticity action per image) and aggregate upstream.
# • Call even if EXIF may be stripped (social apps); summaries remain useful.
# • It is a PLANNING ERROR to produce a tool plan for an image-based claim that
#   does not include this tool for every available image.
#
# OUTPUT CONTRACT
# -----------------------------------------------------------------------------
# • MetadataEvidenceResults.metadata_status ∈ {
#     "incomplete", "complete_original", "complete_manipulated", "unknown"
#   }
# • MetadataEvidenceResults.summary: 120–250 words, concise & neutral (NO verdict).
# • Optional structured fields: timeline, geo, key_metadata, inconsistencies,
#   editing_signals, provenance_hints, checks.
# • MetadataEvidenceResults.is_useful() → True iff a non-empty summary is present.
# • MetadataEvidenceResults.error: diagnostic string on failure (never fatal).
#
# ROBUSTNESS
# -----------------------------------------------------------------------------
# • _perform NEVER raises; non-useful results are returned on failure.
# • If exiftool isn’t available or EXIF is absent, the tool still runs:
#     - OpenAI JSON-mode vision if configured, else
#     - a local rule-based summarizer (fallback).
#
# PROMPTING HINTS (for LLM planners)
# -----------------------------------------------------------------------------
# • For ANY image-based claim, you MUST:
#     - Create a VerifyImageAuthenticity action for EACH image, and
#     - Route it through this metadata_auth_verifier tool.
# • Treat MetadataEvidenceResults as HIGH-PRIORITY evidence for the Judge:
#     - Always pass its summary, timeline, geo, key_metadata, inconsistencies,
#       and checks to downstream judging/verdict steps.
#     - If strong inconsistencies are reported (e.g., impossible device/year,
#       MCC vs GPS mismatch, timezone vs location mismatch, clear editing
#       software on “raw/un-edited” claims), you MUST surface them explicitly
#       in your reasoning plan and treat them as important signals.
# • Do NOT treat its output as an authenticity verdict; combine with other tools
#   (e.g., web search, deepfake detector) if a verdict is required.
# • Failing to call this tool when images are present, or ignoring its outputs
#   when forming a verdict, is considered incorrect planner behavior.
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, cast
from pathlib import Path
import json
import base64
import mimetypes
import os
import re
import tempfile
import yaml
import math
from datetime import datetime, timedelta  # NOTE: timedelta added for TZ math

from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject

# External: OpenAI + ExifTool wrapper + PIL via EZMM
try:
    from openai import OpenAI  # pip install openai
except Exception:
    OpenAI = None

try:
    from exiftool import ExifToolHelper  # pip install exiftool (and system exiftool)
    _EXIFTOOL_AVAILABLE = True
except Exception:
    _EXIFTOOL_AVAILABLE = False

from ezmm import Image, MultimodalSequence
from PIL.Image import Image as PILImage

from defame.common import Results, Action, logger
from defame.evidence_retrieval.tools.tool import Tool


# =========================
# ACTION (robust path handling)
# =========================

class VerifyImageAuthenticity(Action):
    """
    ACTION: Request a metadata evidence summary for a single image using metadata + pixels.
    Lightweight intent object; heavy lifting happens in MetadataAuthVerifier._perform().

    LLM USAGE:
    ----------
    • Construct this action whenever (and only when) the current claim/evidence includes
      an IMAGE. This is MANDATORY for all image-based claims.
    • For each image in the bundle, create exactly one VerifyImageAuthenticity action.
    • Provide image as a path/URL/registry reference resolvable by ezmm.Image.
    • One action per image (planner aggregates results across images).
    • This action does NOT request a real/fake decision; it asks for a neutral evidence
      summary that will be treated as important for the final verdict.
    • The result of this action can be used in interaction with other tools' results like the searcher . compare well
      the found dates and times in both tools .


    """

    name = "verify_image_authenticity"
    requires_image = True  # ← Ensures the planner/runner only uses this when an image exists.

    @staticmethod
    def _looks_like_path(s: str) -> bool:
        s = s.strip()
        if re.match(r"^[A-Za-z]:[\\/]", s):  # Windows drive
            return True
        if s.startswith("\\\\"):  # UNC
            return True
        if s.startswith(("/", "\\", "./", "../")):
            return True
        if os.path.sep in s:
            return True
        if "/" in s:
            return True
        return False

    @staticmethod
    def _normalize_ref(s: str) -> str:
        s = str(s).strip()
        if re.fullmatch(r"<image:\d+>", s):
            return s
        if re.fullmatch(r"image:\d+", s):
            return f"<{s}>"
        if re.fullmatch(r"\d+", s):
            return f"<image:{s}>"
        return s  # URL or other ref

    def __init__(self, image: str):
        # Persist constructor args for reproducibility/debugging across the pipeline.
        self._save_parameters(locals())

        file_path = None
        try:
            if self._looks_like_path(image):
                p = Path(image)
                if p.exists():
                    file_path = str(p.resolve())
        except Exception:
            file_path = None

        if file_path:
            # Important: give ezmm a SAFE registry reference so it doesn't parse the filesystem path.
            ref = "<image:0>"
            self.image = Image(reference=ref, file_path=file_path)
        else:
            ref = self._normalize_ref(image)
            self.image = Image(reference=ref)

    def __str__(self):
        return f'{self.name}({self.image.reference})'

    def __eq__(self, other):
        return isinstance(other, VerifyImageAuthenticity) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


# =========================
# RESULTS
# =========================

@dataclass
class MetadataEvidenceResults(Results):
    """
    RESULT TYPE for metadata-based evidence summarization (no authenticity verdict).

    This object is intended to be treated as HIGH-PRIORITY evidence by downstream
    reasoning / judging components:
    - The Judge SHOULD always read and consider the summary, timeline, geo,
      inconsistencies, and checks when forming a verdict.
    - Strong inconsistencies (e.g., device-year vs capture year, MCC vs GPS,
      timezone vs location) are especially important for the final decision.
    """

    text: str = field(init=False)

    # Core outputs
    metadata_status: Optional[str] = None  # "incomplete" | "complete_original" | "complete_manipulated" | "unknown"
    summary: Optional[str] = None          # concise, neutral; 120–250 words target

    # Structured extras
    timeline: Optional[List[Dict[str, str]]] = None  # [{event, field, value, tz?, source}]
    geo: Optional[Dict[str, Any]] = None             # {"gps_lat","gps_lon","gps_alt"?, "place"?}
    key_metadata: Optional[List[Dict[str, str]]] = None
    inconsistencies: Optional[List[Dict[str, str]]] = None  # [{field, issue, evidence}]
    editing_signals: Optional[List[str]] = None
    provenance_hints: Optional[List[str]] = None
    checks: Optional[List[Dict[str, Any]]] = None    # optional atomic checks

    # Diagnostics
    error: Optional[str] = None

    def is_useful(self) -> Optional[bool]:
        return bool(self.summary and self.summary.strip())

    def __str__(self):
        header = f"Image Metadata Evidence\nMetadata status: {self.metadata_status or 'unknown'}"
        if not self.summary:
            return header + "\nSummary: N/A"

        lines = [header, "", "Summary:", self.summary.strip()]

        if self.timeline:
            lines.append("\nTimeline:")
            for t in self.timeline[:6]:
                ev = t.get("event", "event")
                fld = t.get("field", "field")
                val = t.get("value", "value")
                tz = t.get("tz", "")
                src = t.get("source", "")
                tail = f" ({fld}{', '+tz if tz else ''}{'; '+src if src else ''})"
                lines.append(f"• {ev}: {val}{tail}")

        if self.geo:
            geo_bits = []
            if (
                "gps_lat" in self.geo
                and "gps_lon" in self.geo
                and self.geo.get("gps_lat") not in ("", None)
                and self.geo.get("gps_lon") not in ("", None)
            ):
                geo_bits.append(f"coords={self.geo['gps_lat']},{self.geo['gps_lon']}")
            if "gps_alt" in self.geo and self.geo.get("gps_alt") not in (None, ""):
                geo_bits.append(f"alt={self.geo['gps_alt']}")
            if "place" in self.geo and self.geo.get("place"):
                geo_bits.append(f"place={self.geo['place']}")
            if geo_bits:
                lines.append("\nGeo: " + " | ".join(geo_bits))

        if self.inconsistencies:
            lines.append("\nInconsistencies:")
            for inc in self.inconsistencies[:6]:
                f = inc.get("field", "field?")
                i = inc.get("issue", "issue?")
                e = inc.get("evidence", "")
                lines.append(f"• {f}: {i}" + (f" — {e}" if e else ""))

        if self.editing_signals:
            lines.append("\nEditing signals:")
            for s in self.editing_signals[:6]:
                lines.append(f"• {s}")

        if self.provenance_hints:
            lines.append("\nProvenance hints:")
            for s in self.provenance_hints[:6]:
                lines.append(f"• {s}")

        return "\n".join(lines)

    def __post_init__(self):
        self.text = str(self)


class MetadataAuthVerifier(Tool):
    """
    TOOL WRAPPER around: ExifTool (metadata extraction) + OpenAI JSON-mode vision,
    with a local rule-based fallback summarizer that also detects contradictions.

    FUNCTION:
    ---------
    • Converts VerifyImageAuthenticity actions into a vision+metadata analysis call.
    • Returns MetadataEvidenceResults with a neutral, structured evidence summary.
    • The output is designed to be a PRIMARY source of evidence about device,
      time, place, and potential manipulation cues for the Judge.

    FAILURE POLICY:
    ---------------
    • Never raises from _perform. On any error, returns non-useful result with error set.

    WHEN THE LLM SHOULD PICK THIS TOOL:
    -----------------------------------
    • Whenever the current claim includes an IMAGE, this tool MUST be used.
      It is not optional: for each image you must create one VerifyImageAuthenticity
      action and route it through this tool.
    • Use the resulting MetadataEvidenceResults as important evidence along with
      text search, reverse image search, and any deepfake detectors when forming
      the final verdict.
    • This tool does NOT decide authenticity. Combine with other tools if needed.
    """

    name = "metadata_auth_verifier"
    actions = [VerifyImageAuthenticity]
    summarize = False  # concise .text already provided

    _SYSTEM_INSTRUCTIONS = """
            You are a careful image-metadata analyst. You do NOT decide authenticity.

            You receive: (a) the image itself, and (b) its extracted metadata as JSON from ExifTool.

            Your tasks:

            0) GLOBAL FACT-CHECK PERSPECTIVE:
               Your primary goal is to scan ALL available metadata fields (standard and non-standard)
               and surface any details that could help a downstream fact-checker later.
               This includes, but is not limited to:
                 • capture time and location,
                 • device and software information,
                 • platform/provenance markers (e.g., social network traces, service IDs, FBMD-like tags),
                 • author/creator/credit/copyright/rights fields,
                 • transmission IDs, document IDs, or original reference IDs,
                 • editing / modification history or software pipelines.
               Do NOT restrict yourself to the specific fields listed below. If any unusual or vendor-specific
               tag looks potentially useful for establishing provenance, timeline, source platform, or context,
               you MUST surface it as a key clue in the summary and/or in `provenance_hints`.

            1) Inspect the pixels AND cross-check against metadata to surface:

            • CAPTURE & DEVICE: camera make/model, lens, exposure settings, color space, file type/size, dimensions.
              - When possible, infer the device type (e.g., smartphone vs DSLR vs mirrorless).
              - Use your world knowledge about common camera and smartphone models (e.g., iPhone generations, Galaxy S-series,
                major Nikon/Canon/Sony models) to reason about when a device first became available.
              - If the reported capture date (DateTimeOriginal) is earlier than the earliest plausible release year of the
                declared device model (for example, iPhone 14 Pro with a capture date in 2019, or Galaxy S22 in 2020),
                you MUST treat this as a device–date contradiction.
                • In such a case, add an entry to "inconsistencies" with field like "Model/DateTimeOriginal" and an issue such as
                  "device released after reported capture date", and explain the conflict in the evidence.
                • Also mention this contradiction explicitly in the prose summary.

            • TIME & PLACE: all relevant timestamps (DateTimeOriginal, CreateDate, ModifyDate, SubSec times, timezones) and any GPS
              fields (latitude/longitude/altitude). If present, note them explicitly; if absent, say so.
              - Always pay close attention to Offset Time / OffsetTimeOriginal / OffsetTimeDigitized fields.
              - Check that local time + OffsetTimeOriginal is coherent with the GPS longitude when GPS is present.
              - If country/location tags (IPTC/XMP Country, City, State, etc.) are present, ALWAYS cross-check them against
                the timezone offset.
                For example, offsets around +02:00 are typical for central Europe, not for the United States, whose timezones are
                usually negative offsets (roughly between −11:00 and −03:00). If you see Country/Location = "United States", "USA",
                or similar but OffsetTimeOriginal is +01:00, +02:00, or +03:00, you MUST treat this as a timezone–country mismatch
                and record it in "inconsistencies" with clear evidence.
              - If a MobileCountryCode (MCC) is present, you MUST:
                • Infer which country this MCC belongs to using your own knowledge (e.g., MCC 262 → Germany, 310–316 → USA, etc.).
                • Explicitly mention the MCC and its mapped country in the summary (e.g., “MobileCountryCode MCC=262 (Germany)”).
                • Compare the MCC’s country to:
                    - any GPS coordinates (does the MCC country roughly match where the GPS points?), and
                    - any country/city/location tags in the metadata (IPTC/XMP Country, City, State, etc.).
                  If MCC country obviously disagrees with the apparent location (for example MCC=262/Germany but Country='United States'
                  or city tags clearly in the USA, or GPS in the USA), you MUST:
                    - add an entry in "inconsistencies" (field like "MCC/Country" or "MCC/GPS") describing the mismatch, and
                    - mention this MCC vs location conflict explicitly in the prose summary.

            • PROCESSING PIPELINE: Software tags, rendering/composite flags (HDR/Panorama/MultipleExposure/AI-denoise),
              recompression hints (JPEG subsampling, ICC), and known app pipelines (e.g., social apps).
              - Even when capture EXIF (DateTimeOriginal, GPS, Make/Model) is missing, you MUST inspect and describe any
                remaining technical/provenance metadata such as file type/MIME, image dimensions and megapixels, ICC/color
                profiles, JPEG subsampling, Software tags, file-system timestamps, file path, and OS-specific tags like
                ZoneIdentifier on Windows.
              - Treat platform- or provenance-related fields (e.g., service-specific IDs, “Original Transmission Reference”,
                FBMD-like markers, clearly labeled platform metadata, creator/author/credit fields) as HIGH-VALUE clues when
                present, and surface them in the summary/provenance_hints in a concise, fact-check-relevant way.
              - IMPORTANT: common editing tools such as Photoshop, Lightroom, Snapseed, etc. are widely used for benign
                adjustments (cropping, color correction, export). Their presence ALONE does NOT imply manipulation or deception.
                When you see such software:
                  • Treat it primarily as a provenance / workflow clue.
                  • List it in "editing_signals" and/or "provenance_hints".
                  • Only treat it as a strong manipulation hint when it appears together with OTHER anomalies
                    (e.g., compositing/multiple-exposure flags, inconsistent timestamps, odd resolutions for the device, or
                    a claim that the image is “raw/unedited” that clearly conflicts with the software tags).
                Never imply that the image is manipulated just because Photoshop or similar software appears in the metadata.

            2) Determine a metadata status (NOT an authenticity verdict):

            - "incomplete": metadata missing/minimal (typical of re-encodes with stripped EXIF)
            - "complete_original": appears reasonably complete and internally consistent; may show normal,
              benign editing/export traces (e.g., Photoshop/Lightroom/Snapseed) but no strong contradictions.
            - "complete_manipulated": appears complete but shows evidence that the file is no longer a straight-from-camera
              original, for example explicit compositing/HDR/panorama/AI-resynthesis flags or strong internal inconsistencies
              (e.g., impossible device/year, MCC vs GPS mismatch, timezone vs location discrepancies). The mere presence of
              common editing software (Photoshop, Lightroom, Snapseed, etc.) WITHOUT additional contradictions is NOT enough
              to choose this category; in that case prefer "complete_original" or "unknown" and simply describe the software. 
              that means that the METADATA is manipulated.
            - "unknown": insufficient signal to categorize

            3) Output ONLY a JSON object (no commentary, no authenticity claim) with EXACT keys:

            {
              "metadata_status": "incomplete" | "complete_original" | "complete_manipulated" | "unknown",
              "summary": "<120–250 words, neutral, concise. The FIRST sentence MUST be of the form:
                          'This image was captured on <DATE> at <TIME> using <DEVICE>.'
                          Use 'an unknown date/time/device' if a piece is missing.

                          After that first sentence, explicitly answer the implicit question
                          'Is there any important info in this metadata for future fact-checking?':

                          - Identify and clearly state the 2–5 most fact-check-relevant metadata clues AND their implications
                            for provenance, timeline, source platform, or context (not authenticity).
                            Each important clue MUST mention at least one concrete field name and value.
                            Examples (purely illustrative, not exhaustive):
                              'Key clue: small 735×787 resolution (~0.58 MP) and progressive JPEG → likely downscaled screenshot or web copy, not a camera-original file.'
                              'Key clue: an ICC profile “Color LCD” (Apple Computer Inc., profile date 2025-07-10) → image was created or processed on an Apple display pipeline.'
                              'Key clue: a platform-specific tag or ID (e.g., Original Transmission Reference=...) → suggests the file passed through a particular CMS or social platform.'
                          - Then clearly state which critical capture/location fields are missing
                            (e.g., “Missing fields: no DateTimeOriginal, no GPSLatitude/GPSLongitude, no camera Make/Model tags.”).

                          You MUST NOT use vague statements like 'there is valuable technical data' or
                          'the metadata can still help understand its origins' without naming specific fields.
                          Always be concrete: name the tags (e.g., ImageWidth, ICC Profile Description, ZoneIdentifier,
                          creator/author fields, platform IDs) and describe what they suggest and why they might help a fact-checker.

                          Mention timestamps (and TZ if present) and GPS/location if present.
                          If a MobileCountryCode (MCC) is present, explicitly state the MCC value and the country it corresponds to,
                          and describe any MCC vs location conflicts you detect (e.g., MCC=262 (Germany) but metadata says United States).
                          If you detect a device–date contradiction (capture date earlier than device release), clearly explain it.
                          If you detect a timezone–country mismatch (for example +02:00 with Country='United States'),
                          clearly mention this conflict in the summary as a possible inconsistency.
                          Also describe device/lens (if known), file/container, and notable aspects of the processing pipeline.

                          DO NOT say real/fake or imply authenticity. Your output will be treated as IMPORTANT
                          evidence for the final verdict, so be precise, concrete, and exhaustive within the word budget.>",
              "timeline": [
                {"event": "captured", "field": "DateTimeOriginal", "value": "<ISO 8601 if possible>", "tz": "<offset or empty>", "source": "EXIF"},
                {"event": "file_created", "field": "CreateDate", "value": "...", "tz": "", "source": "EXIF"},
                {"event": "modified", "field": "ModifyDate", "value": "...", "tz": "", "source": "EXIF"}
              ],
              "geo": {
                "gps_lat": "<float or empty>",
                "gps_lon": "<float or empty>",
                "gps_alt": "<float or empty>",
                "place": "<reverse-geocoded name or country tag if present, else empty>"
              },
              "key_metadata": [{"key": "<field>", "value": "<stringified value>"}],
              "inconsistencies": [{"field": "<field>", "issue": "<short description>", "evidence": "<what conflicts with what>"}],
              "editing_signals": ["<Software=...>", "<flags...>"],
              "provenance_hints": ["<device/lens/codec/ICC/author/platform observations>"],
              "checks": [{"category": "<theme>", "verdict": "<pass/fail/unknown>", "evidence": "<brief>"}]
            }

            Guidelines:
            - Summaries must be factual and avoid authenticity claims (no “real/fake/generative” statements).
            - Prefer ISO-8601 timestamps when possible; include timezone if present in metadata.
            - Always reason about:
              • whether the timezone offset you see is plausible for the tagged country/location, and
              • whether the capture date is plausible for the declared device model and its known release period, and
              • whether the MCC’s country aligns with GPS coordinates and country/location tags.
              If not (e.g. +02:00 with United States / USA, a device model that released in 2022 with a 2019 capture date,
              or MCC=262 (Germany) but country tags/GPS clearly indicate the USA),
              you MUST add a corresponding item in "inconsistencies" and reference it from the summary.
            - If a field is missing, put an empty string in JSON and plainly say “not present” in the prose summary.
            - Never claim there is “no metadata” or “no information” if any metadata fields are present. When key EXIF fields
              (DateTimeOriginal, GPS, Make/Model) are missing but other tags exist (file type, size, dimensions, ICC, file-system
              timestamps, ZoneIdentifier, Software, author/credit/copyright, platform IDs), you MUST describe them and explain
              why they may still be useful (for example, indicating a downloaded web copy, a screenshot, an image passed through
              a specific social platform, or a file with known rights/creator).
            - Avoid generic filler sentences like “there is valuable technical data to consider”. Instead, always name specific
              tags and values (e.g., “ImageWidth=736, ImageHeight=981, ~0.72 MP; ICC profile ‘c2’, Little CMS, copyright ‘FB’;
              ZoneIdentifier present; Creator=...”) and briefly state what each implies for provenance or future fact-checking.
            - Manipulation detection: The rules above are MINIMUM checks, not a complete list. You MUST actively look
              for any other plausible signs of metadata tampering or inconsistency, even if they are not explicitly
              described here. If you notice unusual or conflicting patterns (for example, strange or inconsistent date
              formats, impossible resolutions for a known device, partially stripped maker notes with intact dates,
              conflicting values across EXIF/IPTC/XMP, improbable time gaps, or anything that simply looks suspicious),
              you MUST:
                • add an entry in "inconsistencies" describing the issue, and
                • mention this potential manipulation or inconsistency clearly in the prose summary.
              However, do NOT treat the mere presence of common photo-editing software (Photoshop, Lightroom, Snapseed, etc.)
              by itself as evidence of manipulation or deception. Only escalate such software to a serious manipulation concern
              when it coincides with other anomalies or contradicts an explicit claim that the image is “raw” or “unedited”.
            - Your output will be used as a key piece of evidence for a fact-checking verdict; aim for clarity, concreteness,
              and completeness.
            """
    _USER_CHECKLIST = """
            Consider at minimum (but you are encouraged to use ANY other metadata that looks fact-check-relevant):

            - Timestamps: DateTimeOriginal, CreateDate, ModifyDate, SubSecTime*, timezones; monotonic ordering (capture ≤ create ≤ modify).

            - Timezone-specific fields: Offset Time, OffsetTimeOriginal, OffsetTimeDigitized. Check:
              • Whether the offset format itself is valid (e.g. +02:00, −05:00).
              • Whether the offset is plausible for any GPS coordinates (if present).
              • Whether the offset is plausible for the tagged country/location.
                For example, United States / USA usually has negative offsets (roughly −11:00 to −03:00),
                while central European countries like Germany, France, Poland, Spain, Italy often use offsets
                around 0 to +03:00. Treat obvious mismatches (e.g. +02:00 with Country='United States')
                as a timezone–country inconsistency and record them in "inconsistencies".

            - Device vs capture date:
              • Use your general knowledge about device release years (e.g. iPhone 14 Pro ~2022, Galaxy S22 ~2022,
                older iPhones and Android flagships by generation).
              • If DateTimeOriginal is clearly earlier than the device could reasonably exist (for example, a 2018
                capture date with a model that only released in 2022), treat this as an explicit contradiction.
              • Add an "inconsistencies" entry with field like "Model/DateTimeOriginal", issue describing that the
                device released after the reported capture date, and evidence summarizing the model and dates.
              • Mention this conflict in the prose summary.

            - MCC (Mobile Country Code) vs location:
              • If a MobileCountryCode/MCC field is present, infer which country it belongs to (for example,
                some 26x codes map to Germany, 208 to France, 214 to Spain, 310–316 to the USA, 302 to Canada, etc.).
              • Compare this MCC country to:
                  - any country/location tags in the metadata (Country, Country-PrimaryLocationName, XMP:Country,
                    city/state tags that clearly indicate a country), and
                  - any GPS coordinates if present (rough, high-level region match is enough).
              • If MCC country contradicts the apparent location (e.g., MCC maps to Germany but Country='United States'
                or the GPS clearly points to the USA), add an "inconsistencies" entry with field "MCC/Country" or "MCC/GPS",
                describe the mismatch in the issue/evidence fields, and mention this explicitly in the summary.

            - GPS: GPSLatitude, GPSLongitude, GPSAltitude, GPSPosition; if missing, note absence.

            - Device & optics: Make, Model, LensModel/LensInfo; resolution vs typical device output; Orientation.
              • Where possible, infer device type (phone vs larger camera) from Make/Model.
              • Check for contradictions between brand-specific tags (e.g., Apple maker notes, iPhone lens names) and the declared Make/Model.
              • If you know a device’s earliest release year, flag if DateTimeOriginal is earlier than that year.

            - Exposure: ExposureTime/ShutterSpeedValue, FNumber, ISO, FocalLength; plausible combinations.
              • For phones, very long physical focal lengths (e.g., > 15–20 mm) are unusual.
              • For big-sensor cameras, extremely short physical focal lengths (< 4 mm) are unusual.

            - Container/codec: FileType/MIMEType, ImageWidth/Height, BitDepth, Compression, JPEG subsampling, ICC profile.

            - Software & pipeline: Software tag(s), Composite Image/HDR/Panorama/MultipleExposure/SceneType/Processing flags;
              common social-app pipelines that strip EXIF.

            - Platform / provenance / author:
              • Inspect IPTC/XMP or other fields that may encode:
                  - platform or service identifiers,
                  - FBMD-like markers,
                  - Original Transmission Reference or similar IDs,
                  - Creator/Author/By-line/Credit/Copyright fields.
              • Treat these as high-value provenance clues where present.
              • Turn such fields into dedicated 'Key clue:' sentences in the summary when they clearly help identify source,
                rights holder, or platform pipeline.

            - Cross-checks:
              • Timestamps ordering/timezone coherence.
              • Device-release-year vs capture year (if you know when the device first shipped).
              • Device-resolution vs typical device output.
              • ICC/subsampling vs device norms when you can reason about them.
              • Software vs claimed device.
              • GPS plausibility.
              • MobileCountryCode (MCC) vs GPS location and country tags (including city/state information).
              • OffsetTimeOriginal vs GPS longitude and vs any explicit country/location tags.
              • OffsetTimeOriginal vs country/location tags even when GPS is missing. In particular,
                treat offsets around +01:00 to +03:00 with “United States”, “USA”, or “Canada” as suspicious
                and capture this explicitly in the "inconsistencies" array.

            - When key EXIF fields (DateTimeOriginal, GPS, Make/Model) are missing, do NOT treat this as “no metadata”.
              You MUST still inspect and describe:
                • file type/MIME, image width/height and megapixels,
                • file-system timestamps (with timezone if present),
                • file path and OS-specific tags (e.g., ZoneIdentifier on Windows),
                • ICC/color profiles, Software tags,
                • and any author/credit/copyright/platform/provenance IDs that look relevant.
              In the summary, explicitly turn the most important of these into concrete 'Key clue:' statements that name fields
              and values and explain why they might help a fact-checker later.
              Also explicitly list which critical capture/location fields are missing (e.g., 'Missing fields: no DateTimeOriginal,
              no GPSLatitude/GPSLongitude, no camera Make/Model tags.') so the user understands both what is there and what is absent.

            - Finally, do NOT limit manipulation detection to the patterns listed above. If you see any other plausible sign
              of tampering or inconsistency in the metadata or its relationship to the image (for example, odd combinations
              of tags, obviously incorrect units, impossible physical values, or conflicting information across namespaces),
              you MUST:
                • record it in the "inconsistencies" array with a short explanation, and
                • reference it in the summary as a potential manipulation or reliability issue for the metadata.
              It is better to surface a plausible suspicion (clearly labeled as such) than to silently ignore it.
            """

    # Minimal on-device model-release map (used for contradiction checks; extend as needed)
    # NOTE: this is intentionally approximate and only used for "captured before device existed" signals.
    _MODEL_MIN_YEAR: Dict[str, int] = {
        # iPhone
        "iphone 6": 2014,
        "iphone 6s": 2015,
        "iphone 7": 2016,
        "iphone 8": 2017,
        "iphone x": 2017,
        "iphone xr": 2018,
        "iphone xs": 2018,
        "iphone 11": 2019,
        "iphone 12": 2020,
        "iphone 13": 2021,
        "iphone 14": 2022,
        "iphone 15": 2023,
        "iphone 16": 2024,
        # Some common Android flagships (very rough)
        "galaxy s9": 2018,
        "galaxy s10": 2019,
        "galaxy s20": 2020,
        "galaxy s21": 2021,
        "galaxy s22": 2022,
        "galaxy s23": 2023,
        "galaxy s24": 2024,
        "pixel 3": 2018,
        "pixel 4": 2019,
        "pixel 5": 2020,
        "pixel 6": 2021,
        "pixel 7": 2022,
        "pixel 8": 2023,
    }

    # MCC → country (only a small subset; treated as hints, not ground truth)
    _MCC_COUNTRY: Dict[int, str] = {
        262: "Germany",
        260: "Poland",
        214: "Spain",
        208: "France",
        222: "Italy",
        204: "Netherlands",
        206: "Belgium",
        234: "United Kingdom",
        235: "United Kingdom",
        310: "United States",
        311: "United States",
        312: "United States",
        313: "United States",
        302: "Canada",
    }

    # Rough country centres and radii for MCC/GPS distance checks (km)
    _COUNTRY_CENTER_RADIUS: Dict[str, Tuple[Tuple[float, float], float]] = {
        "Germany": ((51.0, 10.0), 500.0),
        "Poland": ((52.0, 19.0), 500.0),
        "Spain": ((40.0, -3.5), 700.0),
        "France": ((47.0, 2.0), 700.0),
        "Italy": ((42.5, 12.5), 600.0),
        "Netherlands": ((52.3, 5.5), 350.0),
        "Belgium": ((50.8, 4.5), 300.0),
        "United Kingdom": ((54.0, -2.5), 600.0),
        "United States": ((39.0, -98.0), 3000.0),
        "Canada": ((56.0, -96.0), 3200.0),
    }

    # Very rough timezone ranges per country (hours vs UTC)
    _COUNTRY_TZ_RANGES: Dict[str, Tuple[float, float]] = {
        "Germany": (0.0, 3.0),
        "Poland": (0.0, 3.0),
        "Spain": (-1.0, 2.0),
        "France": (-1.0, 3.0),
        "Italy": (0.0, 3.0),
        "Netherlands": (0.0, 3.0),
        "Belgium": (0.0, 3.0),
        "United Kingdom": (-1.0, 2.0),
        "United States": (-11.0, -3.0),
        "Canada": (-11.0, -3.0),
    }

    # Brand keywords for simple device-brand vs metadata checks
    _BRAND_KEYWORDS: Dict[str, List[str]] = {
        "apple": ["apple", "iphone", "ipad"],
        "samsung": ["samsung", "galaxy"],
        "google": ["google", "pixel"],
        "huawei": ["huawei"],
        "xiaomi": ["xiaomi", "redmi", "mi "],
        "oneplus": ["oneplus"],
        "nokia": ["nokia"],
        "motorola": ["motorola"],
        "lg": ["lg "],
        "nikon": ["nikon", "d3", "d4", "d5", "d6", "d7", "d8", "coolpix"],
        "canon": ["canon", "eos", "ixus", "powershot"],
        "sony": ["sony", "alpha", "ilce", "a7", "a9"],
        "fujifilm": ["fujifilm", "fuji", "gfx", "x-t", "x-pro"],
        "panasonic": ["panasonic", "lumix", "gh", "g9", "s1"],
        "olympus": ["olympus"],
        "leica": ["leica"],
        "pentax": ["pentax"],
    }

    _SMARTPHONE_BRANDS = {
        "apple", "samsung", "google", "huawei", "xiaomi", "oneplus",
        "nokia", "motorola", "lg"
    }

    _LARGE_CAMERA_BRANDS = {
        "nikon", "canon", "sony", "fujifilm", "panasonic",
        "olympus", "leica", "pentax"
    }

    # ---------- NEW helper methods for TZ/GPS robustness ----------

    @staticmethod
    def _parse_offset_to_hours(offset_str: Optional[str]) -> Optional[float]:
        """'+02:00' -> 2.0 ; '-07:30' -> -7.5 ; returns None if not parsable."""
        if not offset_str:
            return None
        m = re.fullmatch(r'([+-])(\d{2}):?(\d{2})', offset_str.strip())
        if not m:
            return None
        sign = -1 if m.group(1) == '-' else 1
        hh = int(m.group(2))
        mm = int(m.group(3))
        if hh > 14 or mm > 59:
            return None
        return sign * (hh + mm / 60.0)

    @staticmethod
    def _approx_offset_from_longitude(lon: float) -> float:
        """Very rough expected local offset from longitude (each 15° ≈ 1 hour)."""
        lon = max(-180.0, min(180.0, float(lon)))
        return round(lon / 15.0)

    @staticmethod
    def _utc_from_local_iso(iso_like: str, offset_hours: Optional[float]) -> Optional[datetime]:
        """Convert 'YYYY:MM:DD HH:MM:SS[.sss]' with known offset_hours to UTC datetime."""
        if not iso_like or offset_hours is None:
            return None
        try:
            ts = iso_like.replace(" ", "T")
            ts = re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", ts)
            ts = re.sub(r"([+-]\d{2}:?\d{2})$", "", ts).rstrip()
            ts = re.sub(r"\.\d+$", "", ts)
            dt = datetime.fromisoformat(ts)
            return dt - timedelta(hours=offset_hours)
        except Exception:
            return None

    @staticmethod
    def _close_enough(a: Optional[datetime], b: Optional[datetime], minutes: int = 30) -> Optional[bool]:
        if a is None or b is None:
            return None
        delta = abs((a - b).total_seconds()) / 60.0
        return delta <= minutes

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two points on Earth in km."""
        r = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = phi2 - phi1
        dl = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    @classmethod
    def _gps_far_from_country(cls, country: str, lat: float, lon: float) -> Optional[bool]:
        """Return True if GPS is far from typical country centre, False if inside radius, None if unknown."""
        entry = cls._COUNTRY_CENTER_RADIUS.get(country)
        if not entry:
            return None
        (clat, clon), radius = entry
        d = cls._haversine_km(clat, clon, float(lat), float(lon))
        return d > radius

    @classmethod
    def _country_tz_range(cls, country: str) -> Optional[Tuple[float, float]]:
        return cls._COUNTRY_TZ_RANGES.get(country)

    @classmethod
    def _mcc_country(cls, mcc: int) -> Optional[str]:
        return cls._MCC_COUNTRY.get(mcc)

    @staticmethod
    def _normalize_country(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        n = str(name).strip().lower()

        mapping = {
            "germany": "Germany",
            "deutschland": "Germany",
            "de": "Germany",
            "poland": "Poland",
            "polska": "Poland",
            "pl": "Poland",
            "spain": "Spain",
            "españa": "Spain",
            "es": "Spain",
            "france": "France",
            "fr": "France",
            "italy": "Italy",
            "italia": "Italy",
            "it": "Italy",
            "netherlands": "Netherlands",
            "holland": "Netherlands",
            "nl": "Netherlands",
            "belgium": "Belgium",
            "belgië": "Belgium",
            "belgie": "Belgium",
            "be": "Belgium",
            "united kingdom": "United Kingdom",
            "uk": "United Kingdom",
            "great britain": "United Kingdom",
            "gb": "United Kingdom",
            "united states": "United States",
            "united states of america": "United States",
            "usa": "United States",
            "us": "United States",
            "canada": "Canada",
            "ca": "Canada",
        }

        if n in mapping:
            return mapping[n]

        # handle some "Country, State" formats; pick first token
        first = re.split(r"[,/]", n)[0].strip()
        return mapping.get(first, None)

    @staticmethod
    def _extract_country(md: Dict[str, Any]) -> Optional[str]:
        """Try to recover a country label from IPTC/XMP-like tags."""
        country_keys = [
            "IPTC:Country-PrimaryLocationName",
            "Country-PrimaryLocationName",
            "XMP:Country",
            "Country",
            "CountryName",
            "CountryCode",
            "LocationShownCountryName",
        ]
        for k in country_keys:
            v = md.get(k)
            if v not in (None, "", " "):
                return str(v).strip()
        # fallback: any key containing 'country'
        for k, v in md.items():
            if "country" in k.lower() and v not in (None, "", " "):
                return str(v).strip()
        return None

    @classmethod
    def _detect_brands_in_string(cls, s: str) -> List[str]:
        s_low = s.lower()
        brands = []
        for brand, kws in cls._BRAND_KEYWORDS.items():
            if any(kw in s_low for kw in kws):
                brands.append(brand)
        return list(dict.fromkeys(brands))

    @classmethod
    def _detect_declared_brands(cls, make: Optional[str], model: Optional[str]) -> List[str]:
        base = " ".join([str(x) for x in [make, model] if x])
        if not base:
            return []
        return cls._detect_brands_in_string(base)

    @classmethod
    def _detect_metadata_brands(
        cls,
        md: Dict[str, Any],
        lens_model: Optional[str],
        software: Optional[str],
    ) -> List[str]:
        parts = []
        for k, v in md.items():
            kl = k.lower()
            if any(
                key in kl
                for key in [
                    "lens", "maker", "software", "device", "camera",
                    "creator", "history", "apple", "nikon", "canon",
                    "samsung", "huawei", "xiaomi", "pixel"
                ]
            ):
                parts.append(str(v))
        if lens_model:
            parts.append(lens_model)
        if software:
            parts.append(software)
        if not parts:
            return []
        blob = " ".join(parts)
        return cls._detect_brands_in_string(blob)

    @classmethod
    def _device_category_from_brands(cls, brands: List[str]) -> Optional[str]:
        if not brands:
            return None
        bset = set(brands)
        if bset & cls._SMARTPHONE_BRANDS:
            return "phone"
        if bset & cls._LARGE_CAMERA_BRANDS:
            return "camera"
        return None

    # ---------- NEW: prefix helper to enforce "captured on DATE at TIME using DEVICE" ----------

    @staticmethod
    def _build_capture_prefix(dto: Optional[str],
                              tz: Optional[str],
                              make: Optional[str],
                              model: Optional[str]) -> str:
        """
        Build the standardized first sentence:
        'This image was captured on DATE at TIME using DEVICE.'
        Falls back to 'an unknown date/time/device' if needed.
        """
        # DEVICE
        dev = "an unknown device"
        if make or model:
            dev = " ".join([s for s in [make, model] if s]).strip()

        # Defaults
        date_str = "an unknown date"
        time_str = "an unknown time"
        tz_str = ""

        if dto:
            # Try EXIF style: 2025:10:01 21:46:10+02:00
            m = re.match(
                r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})([+-]\d{2}:?\d{2})?$",
                dto.strip()
            )
            if m:
                date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                time_str = f"{m.group(4)}:{m.group(5)}:{m.group(6)}"
                if not tz and m.group(7):
                    tz = m.group(7)
            else:
                # Try ISO-ish: 2025-10-01T21:46:10+02:00
                m2 = re.match(
                    r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})([+-]\d{2}:?\d{2})?$",
                    dto.strip()
                )
                if m2:
                    date_str = m2.group(1)
                    time_str = m2.group(2)
                    if not tz and m2.group(3):
                        tz = m2.group(3)
                else:
                    # Fallback: whole dto as date-ish string
                    date_str = dto.strip()

        if tz:
            tz_clean = tz
            if ":" not in tz_clean and len(tz_clean) == 5:
                tz_clean = f"{tz_clean[:3]}:{tz_clean[3:]}"
            tz_str = f" {tz_clean}"

        return f"This image was captured on {date_str} at {time_str}{tz_str} using {dev}."

    @classmethod
    def _ensure_prefix_with_capture(
        cls,
        summary: str,
        dto: Optional[str],
        tz: Optional[str],
        make: Optional[str],
        model: Optional[str],
    ) -> str:
        """
        Ensure the summary starts with 'This image was captured on ...'.
        If it already does, leave it. Otherwise, prepend the prefix.
        """
        if summary is None:
            summary = ""
        stripped = summary.lstrip()
        if stripped.lower().startswith("this image was captured on"):
            return summary

        prefix = cls._build_capture_prefix(dto, tz, make, model)
        if summary:
            return prefix + " " + summary
        return prefix

    # -------------------------------------------------------------

    def __init__(
        self,
        model: str = "gpt-4o",
        timeout: float = 120.0,
        api_key: Optional[str] = None,
        max_meta_len: int = 2000,
        config_path: str = "config/api_keys.yaml",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.timeout = float(timeout)
        self.max_meta_len = int(max_meta_len)
        self.api_key = (
            api_key
            or self._load_api_key_from_yaml(config_path)
            or os.environ.get("OPENAI_API_KEY")
        )

        if not self.api_key:
            logger.info(
                "[metadata_evidence] OpenAI API key not found; will use local fallback summarizer."
            )

    def _perform(self, action: VerifyImageAuthenticity) -> MetadataEvidenceResults:
        logger.info("[metadata_evidence] Starting metadata evidence summarization...")

        try:
            img_path = self._ensure_local_path(action.image)
            if img_path is None or not Path(img_path).exists():
                msg = "Could not resolve a local path for the provided image."
                logger.warning(f"[metadata_evidence] {msg}")
                return MetadataEvidenceResults(error=msg)

            metadata: Dict[str, Any] = {}
            if _EXIFTOOL_AVAILABLE:
                try:
                    with ExifToolHelper() as et:
                        md_list = et.get_metadata([str(img_path)])
                        metadata = md_list[0] if md_list else {}
                except Exception as ex:
                    logger.warning(
                        f"[metadata_evidence] exiftool failed: {ex} (continuing without EXIF)"
                    )
            else:
                logger.warning(
                    "[metadata_evidence] exiftool module not available; continuing without EXIF"
                )

            metadata = self._slim_metadata(metadata, self.max_meta_len)
            metadata_json = json.dumps(metadata, ensure_ascii=False)

            # Basic fields needed to enforce prefix regardless of analysis path
            make = self._first_nonempty(metadata, ["Make", "Exif:Make", "QuickTime:Make"])
            model = self._first_nonempty(metadata, ["Model", "Exif:Model", "QuickTime:Model"])
            dto = self._first_nonempty(
                metadata, ["DateTimeOriginal", "EXIF:DateTimeOriginal", "QuickTime:CreateDate"]
            )
            cdate = self._first_nonempty(
                metadata, ["CreateDate", "EXIF:CreateDate", "File:FileCreateDate"]
            )
            mdate = self._first_nonempty(
                metadata, ["ModifyDate", "EXIF:ModifyDate", "File:FileModifyDate"]
            )

            # NEW: also consider OffsetTime* tags for timezone hints
            offset_raw = self._first_nonempty(
                metadata,
                [
                    "OffsetTimeOriginal",
                    "EXIF:OffsetTimeOriginal",
                    "OffsetTimeDigitized",
                    "EXIF:OffsetTimeDigitized",
                    "OffsetTime",
                    "EXIF:OffsetTime",
                ],
            )

            tz_hint = (
                self._extract_tz(dto)
                or self._extract_tz(cdate)
                or self._extract_tz(mdate)
                or self._extract_tz(offset_raw)
                or offset_raw
            )

            # Preferred path: OpenAI JSON-mode vision
            if self.api_key and OpenAI is not None:
                try:
                    data_url = self._to_data_url(Path(img_path))
                    client = OpenAI(api_key=self.api_key)
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=cast(
                            List[ChatCompletionMessageParam],
                            [
                                {"role": "system", "content": self._SYSTEM_INSTRUCTIONS},
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": self._USER_CHECKLIST
                                                    + "\n\nHere is the metadata JSON:\n"
                                                    + metadata_json,
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": data_url},
                                        },
                                    ],
                                },
                            ],
                        ),
                        response_format=cast(ResponseFormatJSONObject, {"type": "json_object"}),
                        timeout=self.timeout,
                    )
                    result_json = self._parse_json_strict(
                        resp.choices[0].message.content
                    )

                    raw_summary = result_json.get("summary") or ""
                    timeline = result_json.get("timeline")
                    summary_with_prefix = self._ensure_prefix_with_capture(
                        raw_summary, dto, tz_hint, make, model
                    )

                    return MetadataEvidenceResults(
                        metadata_status=result_json.get("metadata_status") or "unknown",
                        summary=summary_with_prefix,
                        timeline=timeline,
                        geo=result_json.get("geo"),
                        key_metadata=result_json.get("key_metadata"),
                        inconsistencies=result_json.get("inconsistencies"),
                        editing_signals=result_json.get("editing_signals"),
                        provenance_hints=result_json.get("provenance_hints"),
                        checks=result_json.get("checks"),
                    )
                except Exception as api_err:
                    logger.warning(
                        f"[metadata_evidence] OpenAI call failed; falling back locally: {api_err}"
                    )

            # Fallback: local rule-based summarizer
            (
                timeline,
                geo,
                key_metadata,
                editing_signals,
                provenance_hints,
                inconsistencies,
                checks,
                meta_status,
                summary,
            ) = self._local_analyze(metadata, img_path)

            return MetadataEvidenceResults(
                metadata_status=meta_status,
                summary=summary,
                timeline=timeline,
                geo=geo,
                key_metadata=key_metadata,
                inconsistencies=inconsistencies,
                editing_signals=editing_signals,
                provenance_hints=provenance_hints,
                checks=checks,
            )

        except Exception as e:
            logger.warning(f"[metadata_evidence] failed with error: {e}")
            return MetadataEvidenceResults(error=str(e))

    def _summarize(self, result: MetadataEvidenceResults, **kwargs) -> Optional[MultimodalSequence]:
        if not result.is_useful():
            return MultimodalSequence(
                "Metadata evidence summary unavailable or not applicable."
            )

        blocks = [
            f"Metadata status: {result.metadata_status or 'unknown'}",
            result.summary or "",
        ]

        if result.timeline:
            tl = "\n".join(
                f"- {t.get('event','event')}: {t.get('value','')}"
                f" ({t.get('field','')}"
                f"{', '+t.get('tz','') if t.get('tz') else ''}"
                f"{'; '+t.get('source','') if t.get('source') else ''})"
                for t in result.timeline[:5]
            )
            blocks.append("\nTimeline:\n" + tl)

        if result.geo and (
            result.geo.get("gps_lat") not in (None, "")
            and result.geo.get("gps_lon") not in (None, "")
        ):
            geo_line = f"coords={result.geo.get('gps_lat')},{result.geo.get('gps_lon')}"
            if result.geo.get("gps_alt") not in (None, ""):
                geo_line += f" | alt={result.geo.get('gps_alt')}"
            if result.geo.get("place"):
                geo_line += f" | place={result.geo.get('place')}"
            blocks.append("\nGeo:\n- " + geo_line)

        if result.inconsistencies:
            inc_lines = "\n".join(
                f"- {inc.get('field', 'field?')}: {inc.get('issue', '')}"
                + (
                    f" — {inc.get('evidence', '')}"
                    if inc.get("evidence")
                    else ""
                )
                for inc in result.inconsistencies[:5]
            )
            blocks.append("\nKey inconsistencies:\n" + inc_lines)

        if result.editing_signals:
            sig_lines = "\n".join(f"- {s}" for s in result.editing_signals[:5])
            blocks.append("\nEditing signals:\n" + sig_lines)

        if result.provenance_hints:
            prv_lines = "\n".join(f"- {s}" for s in result.provenance_hints[:5])
            blocks.append("\nProvenance hints:\n" + prv_lines)

        return MultimodalSequence(
            "Image Metadata Evidence\n" + "\n\n".join(b for b in blocks if b.strip())
        )

    def _local_analyze(
        self,
        md: Dict[str, Any],
        img_path: Path,
    ) -> Tuple[
        List[Dict[str, str]],
        Dict[str, Any],
        List[Dict[str, str]],
        List[str],
        List[str],
        List[Dict[str, str]],
        List[Dict[str, Any]],
        str,
        str,
    ]:
        make = self._first_nonempty(md, ["Make", "Exif:Make", "QuickTime:Make"])
        model = self._first_nonempty(md, ["Model", "Exif:Model", "QuickTime:Model"])
        software = self._first_nonempty(md, ["Software", "Exif:Software", "QuickTime:Software"])
        orientation = self._first_nonempty(md, ["Orientation", "Exif:Orientation"])

        mime = self._first_nonempty(md, ["MIMEType", "File:MIMEType"])
        file_type = self._first_nonempty(md, ["FileType", "File:FileType"])

        width = self._first_nonempty(
            md, ["ImageWidth", "EXIF:ExifImageWidth", "File:ImageWidth"]
        )
        height = self._first_nonempty(
            md, ["ImageHeight", "EXIF:ExifImageHeight", "File:ImageHeight"]
        )
        color_space = self._first_nonempty(md, ["ColorSpace", "EXIF:ColorSpace"])
        profile = self._first_nonempty(md, ["ICCProfileName", "ProfileDescription"])

        exposure_time = self._first_nonempty(
            md, ["ExposureTime", "ShutterSpeedValue"]
        )
        fnumber = self._first_nonempty(md, ["FNumber", "ApertureValue"])
        iso = self._first_nonempty(md, ["ISO", "PhotographicSensitivity"])

        # Separate physical focal length vs 35mm-equivalent
        focal_real_raw = self._first_nonempty(md, ["FocalLength"])
        focal35_raw = self._first_nonempty(md, ["FocalLengthIn35mmFormat"])
        focal = focal_real_raw or focal35_raw

        focal_mm = self._num_or_empty(focal_real_raw) if focal_real_raw else None
        focal35_mm = self._num_or_empty(focal35_raw) if focal35_raw else None

        lens_model = self._first_nonempty(
            md, ["LensModel", "EXIF:LensModel", "QuickTime:LensModel"]
        )

        dto = self._first_nonempty(
            md, ["DateTimeOriginal", "EXIF:DateTimeOriginal", "QuickTime:CreateDate"]
        )
        cdate = self._first_nonempty(
            md, ["CreateDate", "EXIF:CreateDate", "File:FileCreateDate"]
        )
        mdate = self._first_nonempty(
            md, ["ModifyDate", "EXIF:ModifyDate", "File:FileModifyDate"]
        )

        # NEW: also look at OffsetTime* tags for timezone hints
        offset_raw = self._first_nonempty(
            md,
            [
                "OffsetTimeOriginal",
                "EXIF:OffsetTimeOriginal",
                "OffsetTimeDigitized",
                "EXIF:OffsetTimeDigitized",
                "OffsetTime",
                "EXIF:OffsetTime",
            ],
        )

        tz_hint = (
            self._extract_tz(dto)
            or self._extract_tz(cdate)
            or self._extract_tz(mdate)
            or self._extract_tz(offset_raw)
            or offset_raw
        )

        gps_lat = self._num_or_empty(
            self._first_nonempty(md, ["GPSLatitude", "Composite:GPSLatitude"])
        )
        gps_lon = self._num_or_empty(
            self._first_nonempty(md, ["GPSLongitude", "Composite:GPSLongitude"])
        )
        gps_alt = self._num_or_empty(
            self._first_nonempty(md, ["GPSAltitude", "Composite:GPSAltitude"])
        )

        country_hint_raw = self._extract_country(md)
        country_hint = self._normalize_country(country_hint_raw)

        geo = {
            "gps_lat": gps_lat if gps_lat is not None else "",
            "gps_lon": gps_lon if gps_lon is not None else "",
            "gps_alt": gps_alt if gps_alt is not None else "",
            "place": country_hint or (country_hint_raw or ""),
        }

        timeline: List[Dict[str, str]] = []
        if dto:
            timeline.append(
                {
                    "event": "captured",
                    "field": "DateTimeOriginal",
                    "value": dto,
                    "tz": tz_hint or "",
                    "source": "EXIF",
                }
            )
        if cdate:
            timeline.append(
                {
                    "event": "file_created",
                    "field": "CreateDate",
                    "value": cdate,
                    "tz": "",
                    "source": "EXIF/FS",
                }
            )
        if mdate:
            timeline.append(
                {
                    "event": "modified",
                    "field": "ModifyDate",
                    "value": mdate,
                    "tz": "",
                    "source": "EXIF/FS",
                }
            )

        key_metadata: List[Dict[str, str]] = []
        for k, v in [
            ("Make", make),
            ("Model", model),
            ("Software", software),
            ("Orientation", orientation),
            ("MIMEType", mime or file_type),
            ("ColorSpace", color_space),
            ("ICCProfile", profile),
            (
                "Dimensions",
                f"{width}x{height}" if width and height else "",
            ),
            ("Exposure", exposure_time),
            ("FNumber", fnumber),
            ("ISO", iso),
            ("FocalLength", focal),
        ]:
            if v not in (None, ""):
                key_metadata.append({"key": k, "value": str(v)})

        # MCC (Mobile Country Code) if present → include in Fact-check-relevant metadata
        mcc_raw = self._first_nonempty(md, ["MobileCountryCode", "MCC"])
        if mcc_raw not in (None, ""):
            key_metadata.append({"key": "MobileCountryCode", "value": str(mcc_raw)})

        editing_signals = self._editing_signals(md, software)

        prov: List[str] = []
        if make or model:
            prov.append(f"Device={make or ''} {model or ''}".strip())
        if file_type or mime:
            prov.append(f"Container={file_type or mime}")
        if profile:
            prov.append(f"ICC={profile}")
        if orientation:
            prov.append(f"Orientation={orientation}")
        provenance_hints = prov

        checks, inconsistencies = self._auto_checks(
            md=md,
            make=make,
            model=model,
            dto=dto,
            cdate=cdate,
            mdate=mdate,
            width=width,
            height=height,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            software=software,
            tz_hint=tz_hint,
            gps_alt=gps_alt,
            country_hint=country_hint,
            lens_model=lens_model,
            focal_mm=focal_mm,
            focal35_mm=focal35_mm,
            mcc_raw=mcc_raw,
        )

        meta_status = self._decide_status(md, editing_signals, checks)

        summary = self._build_summary(
            meta_status,
            make,
            model,
            width,
            height,
            mime or file_type,
            color_space,
            profile,
            dto,
            cdate,
            mdate,
            tz_hint,
            gps_lat,
            gps_lon,
            gps_alt,
            software,
            orientation,
            exposure_time,
            fnumber,
            iso,
            focal,
            inconsistencies,
            editing_signals,
            mcc_raw,  # <-- MCC included in summary / Fact-check-relevant metadata
        )

        return (
            timeline,
            geo,
            key_metadata,
            editing_signals,
            provenance_hints,
            inconsistencies,
            checks,
            meta_status,
            summary,
        )

    def _build_summary(
        self,
        status,
        make,
        model,
        width,
        height,
        mime,
        color_space,
        profile,
        dto,
        cdate,
        mdate,
        tz,
        gps_lat,
        gps_lon,
        gps_alt,
        software,
        orientation,
        exposure_time,
        fnumber,
        iso,
        focal,
        inconsistencies,
        editing_signals,
        mcc_raw: Optional[str] = None,  # <-- MCC available to summary
    ) -> str:
        parts: List[str] = []

        dev = "unknown device"
        if make or model:
            dev = " ".join([s for s in [make, model] if s]).strip()
        dims = f"{width}×{height}" if width and height else "unknown dimensions"
        cont = mime or "unknown container"
        cs = color_space or "unknown color space"

        # NOTE: first sentence ("This image was captured ...") is enforced later
        parts.append(
            f"This image appears to originate from {dev} with {dims} resolution. "
            f"The file container/type is {cont}, and the color space is {cs}"
            + (f" (ICC profile: {profile})." if profile else ".")
        )

        time_bits: List[str] = []
        if dto:
            time_bits.append(f"capture (DateTimeOriginal): {dto}")
        if cdate:
            time_bits.append(f"file creation: {cdate}")
        if mdate:
            time_bits.append(f"last modification: {mdate}")
        if time_bits:
            tz_part = f" Timezone hint: {tz}." if tz else ""
            parts.append("Timestamps observed — " + "; ".join(time_bits) + "." + tz_part)
        else:
            parts.append(
                "No standard EXIF timestamps (DateTimeOriginal/CreateDate/ModifyDate) were found."
            )

        if gps_lat not in ("", None) and gps_lon not in ("", None):
            geo_line = f"GPS coordinates present ({gps_lat}, {gps_lon}"
            if gps_alt not in ("", None):
                geo_line += f", alt {gps_alt}"
            geo_line += ")."
            parts.append(geo_line)
        else:
            parts.append("No GPS coordinates are present in the metadata.")

        # Explicit MCC line with mapped country if possible
        if mcc_raw not in (None, ""):
            mcc_txt = str(mcc_raw)
            mcc_country_str = ""
            mcc_num_match = re.search(r"(\d{3})", mcc_txt)
            if mcc_num_match:
                try:
                    mcc_val = int(mcc_num_match.group(1))
                    mcc_country = self._mcc_country(mcc_val)
                    if mcc_country:
                        mcc_country_str = f"{mcc_val} ({mcc_country})"
                    else:
                        mcc_country_str = str(mcc_val)
                except Exception:
                    mcc_country_str = mcc_txt
            else:
                mcc_country_str = mcc_txt

            if mcc_country_str:
                parts.append(
                    f"MobileCountryCode MCC={mcc_country_str} is present in the metadata."
                )
            else:
                parts.append(
                    "A MobileCountryCode (MCC) value is present in the metadata."
                )

        exp: List[str] = []
        if exposure_time:
            exp.append(f"ExposureTime={exposure_time}")
        if fnumber:
            exp.append(f"FNumber={fnumber}")
        if iso:
            exp.append(f"ISO={iso}")
        if focal:
            exp.append(f"FocalLength={focal}")
        if exp:
            parts.append("Capture parameters: " + ", ".join(exp) + ".")

        if orientation:
            parts.append(f"Orientation reported as {orientation}.")
        if software:
            parts.append(f"Processing software recorded as “{software}”.")
        if editing_signals:
            parts.append("Editing signals: " + "; ".join(editing_signals) + ".")

        if inconsistencies:
            inc_lines = "; ".join(
                [f"{i.get('field','?')}: {i.get('issue','')}" for i in inconsistencies[:5]]
            )
            parts.append("Potential inconsistencies: " + inc_lines + ".")

        # ---- Explicit Fact-check-relevant metadata line including MCC ----
        fact_bits: List[str] = []
        if dto or cdate or mdate:
            fact_bits.append("capture/file timestamps")
        if tz:
            fact_bits.append("timezone offset")
        if gps_lat not in ("", None) and gps_lon not in ("", None):
            fact_bits.append("GPS position")
        if mcc_raw not in (None, ""):
            # Try to enrich with country name if known
            mcc_desc = "MobileCountryCode"
            mcc_num_match2 = re.search(r"(\d{3})", str(mcc_raw))
            if mcc_num_match2:
                try:
                    mcc_val2 = int(mcc_num_match2.group(1))
                    mcc_country2 = self._mcc_country(mcc_val2)
                    if mcc_country2:
                        mcc_desc = f"MobileCountryCode (MCC={mcc_val2}, {mcc_country2})"
                    else:
                        mcc_desc = f"MobileCountryCode (MCC={mcc_val2})"
                except Exception:
                    mcc_desc = f"MobileCountryCode (MCC={mcc_raw})"
            else:
                mcc_desc = f"MobileCountryCode (MCC={mcc_raw})"
            fact_bits.append(mcc_desc)
        if software:
            fact_bits.append("processing software tags")
        if dev != "unknown device":
            fact_bits.append("device/lens identifiers")

        if fact_bits:
            parts.append(
                "Fact-check-relevant metadata includes "
                + ", ".join(fact_bits)
                + "."
            )

        parts.append(
            f"Metadata status: {status}. This summary is informational and does not assert authenticity."
        )

        text = " ".join(parts)
        words = text.split()
        if len(words) > 260:
            text = " ".join(words[:260]) + "…"

        # Enforce: first sentence must be "This image was captured on DATE at TIME using DEVICE."
        text = self._ensure_prefix_with_capture(text, dto, tz, make, model)
        return text

    def _auto_checks(
        self,
        md: Dict[str, Any],
        make: Optional[str],
        model: Optional[str],
        dto: Optional[str],
        cdate: Optional[str],
        mdate: Optional[str],
        width: Optional[str],
        height: Optional[str],
        gps_lat: Optional[float],
        gps_lon: Optional[float],
        software: Optional[str],
        tz_hint: Optional[str] = None,
        gps_alt: Optional[float] = None,
        country_hint: Optional[str] = None,
        lens_model: Optional[str] = None,
        focal_mm: Optional[float] = None,
        focal35_mm: Optional[float] = None,
        mcc_raw: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        checks: List[Dict[str, Any]] = []
        inconsistencies: List[Dict[str, str]] = []

        def _year(s: Optional[str]) -> Optional[int]:
            if not s:
                return None
            m = re.search(r"(\d{4})", s)
            if not m:
                return None
            try:
                y = int(m.group(1))
                if 1990 <= y <= 2100:
                    return y
            except Exception:
                pass
            try:
                dt = datetime.fromisoformat(s.replace(" ", "T").replace(":", "-", 2))
                return dt.year
            except Exception:
                return None

        y_cap, y_create, y_mod = _year(dto), _year(cdate), _year(mdate)

        def _cmp(a, b):
            if a is None or b is None:
                return None
            return -1 if a < b else (0 if a == b else 1)

        # --- timestamp monotonicity ---
        ord1 = _cmp(y_cap, y_create)
        ord2 = _cmp(y_create, y_mod)
        t_ok = True
        if ord1 is not None and ord1 == 1:
            t_ok = False
            inconsistencies.append(
                {
                    "field": "DateTimeOriginal/CreateDate",
                    "issue": "capture after file creation",
                    "evidence": f"{dto} > {cdate}",
                }
            )
        if ord2 is not None and ord2 == 1:
            t_ok = False
            inconsistencies.append(
                {
                    "field": "CreateDate/ModifyDate",
                    "issue": "file created after modification",
                    "evidence": f"{cdate} > {mdate}",
                }
            )
        checks.append(
            {
                "category": "timestamps_order",
                "verdict": "pass" if t_ok else "fail",
                "evidence": "" if t_ok else "non-monotonic year ordering",
            }
        )

        # --- device release year vs capture date ---
        dev_ok = True
        if model:
            ml = model.lower()
            for k, miny in self._MODEL_MIN_YEAR.items():
                if k in ml:
                    if y_cap is not None and y_cap < miny:
                        dev_ok = False
                        inconsistencies.append(
                            {
                                "field": "Model/DateTimeOriginal",
                                "issue": "device released after reported capture year",
                                "evidence": f"Model={model} (≥{miny}) vs capture year={y_cap}",
                            }
                        )
                    break
        checks.append(
            {
                "category": "device_year_vs_capture",
                "verdict": "pass" if dev_ok else "fail",
                "evidence": ""
                if dev_ok
                else "model appears newer than capture date",
            }
        )

        # --- dimensions presence ---
        dim_ok = bool(width and height)
        checks.append(
            {
                "category": "dimensions_present",
                "verdict": "pass" if dim_ok else "unknown",
                "evidence": f"{width}x{height}" if dim_ok else "missing",
            }
        )

        # --- GPS presence consistency ---
        gps_ok = True
        if (gps_lat in ("", None)) ^ (gps_lon in ("", None)):
            gps_ok = False
            inconsistencies.append(
                {
                    "field": "GPS",
                    "issue": "only one coordinate present",
                    "evidence": f"lat={gps_lat}, lon={gps_lon}",
                }
            )
        checks.append(
            {
                "category": "gps_presence",
                "verdict": "pass" if gps_ok else "fail",
                "evidence": "" if gps_ok else "partial GPS",
            }
        )

        # --- editing software flags ---
        soft_flags = self._editing_signals(md, software)
        soft_ok = not soft_flags
        if not soft_ok:
            inconsistencies.append(
                {
                    "field": "Software",
                    "issue": "editing/manipulation software detected",
                    "evidence": "; ".join(soft_flags) if False else "; ".join(soft_flags),
                }
            )
        checks.append(
            {
                "category": "software_editing",
                "verdict": "pass" if soft_ok else "fail",
                "evidence": "" if soft_ok else "; ".join(soft_flags),
            }
        )

        # ---------- NEW: Timezone checks ----------
        tz_ok = True
        tz_hours = self._parse_offset_to_hours(tz_hint) if tz_hint else None
        if tz_hint and tz_hours is None:
            tz_ok = False
            inconsistencies.append(
                {
                    "field": "OffsetTimeOriginal",
                    "issue": "invalid timezone format",
                    "evidence": f"OffsetTimeOriginal={tz_hint}",
                }
            )
        checks.append(
            {
                "category": "tz_format",
                "verdict": "pass" if tz_ok else "fail",
                "evidence": "" if tz_ok else f"bad tz {tz_hint}",
            }
        )

        # TZ vs GPS longitude
        tz_gps_ok = True
        if (
            tz_hours is not None
            and (gps_lat not in ("", None))
            and (gps_lon not in ("", None))
        ):
            approx = self._approx_offset_from_longitude(gps_lon)
            if abs(tz_hours - approx) > 1.0:  # allow ~±1h for DST / borders
                tz_gps_ok = False
                inconsistencies.append(
                    {
                        "field": "OffsetTimeOriginal/GPS",
                        "issue": "timezone offset conflicts with GPS longitude",
                        "evidence": f"Offset={tz_hours:+.0f} vs approx_from_lon={approx:+.0f} (lon={gps_lon})",
                    }
                )
            # coarse region sanity
            if gps_lon <= -60 and not (-10 <= tz_hours <= -4):
                tz_gps_ok = False
                inconsistencies.append(
                    {
                        "field": "OffsetTimeOriginal/GPS",
                        "issue": "US-like longitude with non-American timezone offset",
                        "evidence": f"lon={gps_lon} implies Americas, offset={tz_hint}",
                    }
                )
            if -10 <= gps_lon <= 40 and not (0 <= tz_hours <= 3):
                tz_gps_ok = False
                inconsistencies.append(
                    {
                        "field": "OffsetTimeOriginal/GPS",
                        "issue": "European-like longitude with unusual timezone offset",
                        "evidence": f"lon={gps_lon}, offset={tz_hint}",
                    }
                )
        checks.append(
            {
                "category": "tz_vs_gps",
                "verdict": "pass" if tz_gps_ok else "fail",
                "evidence": "" if tz_gps_ok else "offset/longitude mismatch",
            }
        )

        # TZ vs explicit country/location tags
        tz_country_verdict = "unknown"
        if tz_hours is not None and country_hint:
            rng = self._country_tz_range(country_hint)
            if rng:
                lo, hi = rng
                if tz_hours < lo - 0.5 or tz_hours > hi + 0.5:
                    tz_country_verdict = "fail"
                    inconsistencies.append(
                        {
                            "field": "OffsetTimeOriginal/Country",
                            "issue": "timezone offset conflicts with country/location tags",
                            "evidence": f"Country={country_hint}, offset={tz_hint}",
                        }
                    )
                else:
                    tz_country_verdict = "pass"
        checks.append(
            {
                "category": "tz_vs_country",
                "verdict": tz_country_verdict,
                "evidence": "",
            }
        )

        # GPSTimeStamp (UTC) vs local DateTimeOriginal + offset
        gps_utc_raw = md.get("GPSTimeStamp") or md.get("EXIF:GPSTimeStamp")
        gps_utc_verdict = "unknown"
        if gps_utc_raw and dto and tz_hours is not None:
            try:
                nums = [int(x) for x in re.findall(r"\d+", str(gps_utc_raw))[:3]]
                if len(nums) == 3:
                    h, m, s = nums
                    dto_utc = self._utc_from_local_iso(dto, tz_hours)
                    if dto_utc:
                        gps_utc = dto_utc.replace(
                            hour=h, minute=m, second=s, microsecond=0
                        )
                        ok = self._close_enough(dto_utc, gps_utc, minutes=90)
                        gps_utc_verdict = "pass" if ok else "fail"
                        if not ok:
                            inconsistencies.append(
                                {
                                    "field": "GPSTimeStamp/DateTimeOriginal",
                                    "issue": "UTC GPSTimeStamp diverges from local timestamp+offset",
                                    "evidence": f"DateTimeOriginal={dto} (tz={tz_hint}) vs GPSTimeStamp={gps_utc_raw}",
                                }
                            )
            except Exception:
                gps_utc_verdict = "unknown"
        checks.append(
            {
                "category": "gps_utc_vs_local",
                "verdict": gps_utc_verdict,
                "evidence": "",
            }
        )

        # ---------- MCC vs GPS and Country ----------
        mcc_verdict = "unknown"
        mcc_country: Optional[str] = None
        if mcc_raw not in (None, ""):
            mm = re.search(r"(\d{3})", str(mcc_raw))
            if mm:
                try:
                    mcc_val = int(mm.group(1))
                    mcc_country = self._mcc_country(mcc_val)
                except Exception:
                    mcc_country = None

        if mcc_country and gps_lat not in (None, "") and gps_lon not in (None, ""):
            far = self._gps_far_from_country(mcc_country, gps_lat, gps_lon)
            if far is True:
                mcc_verdict = "fail"
                inconsistencies.append(
                    {
                        "field": "MCC/GPS",
                        "issue": "MCC country inconsistent with GPS coordinates",
                        "evidence": f"MCC={mcc_raw} → {mcc_country} vs GPS=({gps_lat},{gps_lon})",
                    }
                )
            elif far is False:
                mcc_verdict = "pass"
        checks.append(
            {
                "category": "mcc_vs_gps",
                "verdict": mcc_verdict,
                "evidence": "",
            }
        )

        mcc_country_verdict = "unknown"
        if mcc_country and country_hint:
            if mcc_country != country_hint:
                mcc_country_verdict = "fail"
                inconsistencies.append(
                    {
                        "field": "MCC/Country",
                        "issue": "MCC country disagrees with country/location tags",
                        "evidence": f"MCC={mcc_raw} → {mcc_country}, Country={country_hint}",
                    }
                )
            else:
                mcc_country_verdict = "pass"
        checks.append(
            {
                "category": "mcc_vs_country",
                "verdict": mcc_country_verdict,
                "evidence": "",
            }
        )

        # ---------- Device brand vs metadata, and device vs optics ----------
        declared_brands = self._detect_declared_brands(make, model)
        metadata_brands = self._detect_metadata_brands(md, lens_model, software)

        brand_verdict = "unknown"
        if declared_brands and metadata_brands:
            declared_set = set(declared_brands)
            meta_set = set(metadata_brands)
            foreign = [b for b in meta_set if b not in declared_set]
            if foreign:
                brand_verdict = "fail"
                inconsistencies.append(
                    {
                        "field": "Device/Brand",
                        "issue": "brand-specific metadata contradicts declared Make/Model",
                        "evidence": f"Declared={declared_brands}, Metadata={metadata_brands}",
                    }
                )
            else:
                brand_verdict = "pass"
        elif declared_brands:
            brand_verdict = "pass"
        checks.append(
            {
                "category": "device_brand_vs_metadata",
                "verdict": brand_verdict,
                "evidence": "",
            }
        )

        # Device type vs optics (very rough heuristics)
        all_brands = list(dict.fromkeys(declared_brands + metadata_brands))
        dev_cat = self._device_category_from_brands(all_brands)
        optics_verdict = "unknown"
        if dev_cat == "phone" and focal_mm is not None:
            # Physical focal lengths > ~15mm are unusual for phones (tele modules exist but are rare).
            if focal_mm > 15.0:
                optics_verdict = "fail"
                inconsistencies.append(
                    {
                        "field": "Device/FocalLength",
                        "issue": "unusually long physical focal length for a phone-like device",
                        "evidence": f"Device brands={all_brands}, FocalLength={focal_mm} mm",
                    }
                )
            else:
                optics_verdict = "pass"
        elif dev_cat == "camera" and focal_mm is not None:
            # Full-frame / large-sensor cameras rarely have physical focal < 4mm
            if focal_mm < 4.0:
                optics_verdict = "fail"
                inconsistencies.append(
                    {
                        "field": "Device/FocalLength",
                        "issue": "very short physical focal length for a large-sensor camera",
                        "evidence": f"Device brands={all_brands}, FocalLength={focal_mm} mm",
                    }
                )
            else:
                optics_verdict = "pass"

        checks.append(
            {
                "category": "device_vs_optics",
                "verdict": optics_verdict,
                "evidence": "",
            }
        )

        return checks, inconsistencies

    def _editing_signals(self, md: Dict[str, Any], software: Optional[str]) -> List[str]:
        sigs: List[str] = []

        sw = (software or "").lower()
        for tag in ["Software", "HistorySoftwareAgent", "ProcessingSoftware", "Application", "CreatorTool"]:
            val = str(md.get(tag, "") or md.get(f"EXIF:{tag}", "")).lower()
            if val:
                sw = sw + " " + val

        known = [
            "photoshop",
            "lightroom",
            "gimp",
            "affinity",
            "snapseed",
            "picsart",
            "mediatek denoise",
            "topaz",
            "remini",
            "prisma",
            "afterlight",
            "luminar",
            "skylum",
            "pixelmator",
            "canva",
        ]
        for k in known:
            if k in sw:
                sigs.append(f"Software={k}")

        composite_keys = [
            "CompositeImage",
            "HDRImageType",
            "Panorama",
            "ImageProcessing",
            "MultipleExposure",
            "SceneCaptureType",
            "DigitalZoomRatio",
            "AIProcessing",
            "SequentialShot",
        ]
        for ck in composite_keys:
            v = str(md.get(ck, "")).strip()
            if v and v not in ("0", "0 0", "None", "Off", "False"):
                sigs.append(f"{ck}={v}")

        return list(dict.fromkeys(sigs))

    def _decide_status(self, md, editing_signals, checks):
        camera_keys = [
            "Make",
            "Model",
            "DateTimeOriginal",
            "CreateDate",
            "ModifyDate",
            "ISO",
            "FNumber",
            "ExposureTime",
            "FocalLength",
            "Orientation",
            "ColorSpace",
        ]
        present = sum(
            1
            for k in camera_keys
            if any(k2.endswith(k) or k2 == k for k2 in md.keys())
        )

        any_fail = any(ch["verdict"] == "fail" for ch in checks)
        any_tz_warn = any(
            ch["category"]
            in ("tz_vs_gps", "gps_utc_vs_local", "tz_format", "tz_vs_country")
            and ch["verdict"] in ("fail", "unknown")
            for ch in checks
        )

        if present <= 2:
            return "incomplete"

        if editing_signals or any_fail or any_tz_warn:
            # Treat tz/gps/mcc/brand/optics inconsistencies as manipulated/inconclusive (not original).
            return "complete_manipulated"

        if present >= 6:
            return "complete_original"

        return "unknown"

    # ---------- helpers below ----------

    @staticmethod
    def _load_api_key_from_yaml(config_path: str) -> Optional[str]:
        try:
            cfg_file = Path(config_path)
            if not cfg_file.exists():
                alt = Path(__file__).resolve().parent / config_path
                cfg_file = alt if alt.exists() else cfg_file
            if not cfg_file.exists():
                return None
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "openai_api_key" in data:
                return data["openai_api_key"]
            if isinstance(data.get("openai"), dict) and "api_key" in data["openai"]:
                return data["openai"]["api_key"]
            if "open_api_key" in data:
                return data["open_api_key"]
            return None
        except Exception:
            return None

    @staticmethod
    def _ensure_local_path(image_obj: Image) -> Optional[Path]:
        def _get_path_attr(obj: Any, name: str) -> Optional[Path]:
            if not hasattr(obj, name):
                return None
            attr = getattr(obj, name)
            try:
                candidate = attr() if callable(attr) else attr
            except TypeError:
                candidate = attr
            if isinstance(candidate, (str, Path)) and str(candidate):
                return Path(candidate)
            return None

        for candidate_name in ("path", "local_path", "filepath", "file_path", "to_path"):
            p = _get_path_attr(image_obj, candidate_name)
            if p:
                return p

        pil: PILImage = image_obj.image
        if pil is None:
            return None
        tmp = tempfile.NamedTemporaryFile(
            prefix="defame_meta_evidence_", suffix=".jpg", delete=False
        )
        try:
            pil.save(tmp.name, format="JPEG", quality=92)
            return Path(tmp.name)
        finally:
            tmp.close()

    @staticmethod
    def _to_data_url(image_path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(image_path))
        if mime is None:
            mime = "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _slim_metadata(md: Dict[str, Any], maxlen: int = 2000) -> Dict[str, Any]:
        DROP_KEYS_PARTIAL = (
            "ThumbnailImage",
            "ThumbnailLength",
            "ThumbnailOffset",
            "HDRGainCurve",
            "HDRGainCurveSize",
            "RedTRC",
            "GreenTRC",
            "BlueTRC",
            "Red Tone Reproduction Curve",
            "Green Tone Reproduction Curve",
            "Blue Tone Reproduction Curve",
            "APP10:",
        )
        pruned: Dict[str, Any] = {}
        for k, v in md.items():
            if any(part in k for part in DROP_KEYS_PARTIAL):
                continue
            if isinstance(v, str):
                if "Binary data" in v:
                    continue
                if len(v) > maxlen:
                    v = v[:maxlen] + "…"
            pruned[k] = v
        return pruned

    @staticmethod
    def _parse_json_strict(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r"\{(?:[^{}]|(?R))*\}", s, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise

    @staticmethod
    def _first_nonempty(md: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for k in keys:
            v = md.get(k)
            if v in (None, "", " "):
                continue
            return str(v)
        for k in keys:
            for kk, vv in md.items():
                if kk.endswith(k) and vv not in (None, "", " "):
                    return str(vv)
        return None

    @staticmethod
    def _num_or_empty(v) -> Optional[float]:
        if v in (None, "", " "):
            return None
        try:
            return float(str(v).replace(",", "."))
        except Exception:
            return None

    @staticmethod
    def _extract_tz(ts: Optional[str]) -> Optional[str]:
        if not ts:
            return None
        m = re.search(r"([+-]\d{2}:?\d{2})", ts)
        return m.group(1) if m else None
