# =============================================================================
# Deepfake detection tool (Effort / DeepfakeBench integration)
#
# PURPOSE (for LLM planners / tool-using agents)
# -----------------------------------------------------------------------------
# • Use this tool whenever a claim includes (or references) an IMAGE.
# • It returns a scalar probability in [0, 1] that the image is FAKE.
# • The planner should pass this probability to downstream decision steps
#   (e.g., classification / judging). A high score → likely "Fake", low → "Real".
#
# WHEN TO CALL
# -----------------------------------------------------------------------------
# • If a Claim contains an image or an image reference you can resolve to a PIL image.
# • If multiple images exist, run the tool per-image (one action per image) and aggregate.
# • If the detector cannot run (missing weights/config; unreadable image; etc.), the tool
#   returns a non-useful result (fake_probability=None) instead of raising an exception.
#
# OUTPUT CONTRACT
# -----------------------------------------------------------------------------
# • DeepfakeResults.fake_probability: float in [0, 1] (probability image is Fake).
# • DeepfakeResults.label: "Fake" or "Real" via threshold (default 0.5).
# • DeepfakeResults.is_useful(): True iff a usable probability was produced.
# • DeepfakeResults.error: optional diagnostic string (never fatal to the pipeline).
#
# ROBUSTNESS
# -----------------------------------------------------------------------------
# • The tool never raises from _perform; the pipeline keeps going.
# • If the optional landmark model is missing, the detector runs without alignment.
# • If no face is found or upstream code returns None, the tool returns non-useful.
#
# PROMPTING HINTS (LLM)
# -----------------------------------------------------------------------------
# • Prefer close-up, frontal faces for best accuracy (but not strictly required).
# • If the result is borderline near the threshold, avoid definitive claims.
# • If is_useful() is False, skip using this score and consider other tools (e.g., search).
# =============================================================================

from third_party.effort.DeepfakeBench.training.demo import effort_fake_probability
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import torch
from PIL.Image import Image as PILImage
from ezmm import Image, MultimodalSequence

from defame.common import Results, Action, logger
from defame.evidence_retrieval.tools.tool import Tool


class DetectDeepfake(Action):
    """
    ACTION: Request a deepfake probability for a single image.

    LLM USAGE:
    ----------
    • Construct this action if (and only if) the current claim/evidence includes an image.
    • Provide `image` as a path/URL reference resolvable by `ezmm.Image`.
    • This action delegates actual work to the `DeepfakeDetector` tool.

    ATTRIBUTES:
    -----------
    name           : Tool routing key ("detect_deepfake").
    requires_image : Signals to the tool runner that an image MUST be present.

    NOTE:
    -----
    This class is a lightweight “intent” object. The heavy lifting is done in
    DeepfakeDetector._perform().
    """
    name = "detect_deepfake"
    requires_image = True  # ← The planner/runner must attach an image reference.

    def __init__(self, image: str):
        # Persist constructor args for reproducibility/debugging across the pipeline.
        self._save_parameters(locals())
        # Wrap raw reference with EZMM's Image (lazy conversion to PIL when accessed).
        self.image = Image(reference=image)

    def __str__(self):
        return f'{self.name}({self.image.reference})'

    def __eq__(self, other):
        return isinstance(other, DetectDeepfake) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


@dataclass
class DeepfakeResults(Results):
    """
    RESULT TYPE for deepfake detection.

    FIELDS:
    -------
    fake_probability : Optional[float]
        Probability that the image is FAKE (class 1), in [0, 1].
        None → non-useful (e.g., missing assets or detector skipped).

    label : Optional[str]
        "Fake" or "Real" derived from a configured decision threshold (default 0.5).
        If fake_probability is None, label may be None too.

    error : Optional[str]
        Diagnostic message (non-fatal). Present when the tool could not produce a score.

    text : str
        A concise human-readable summary (auto-filled from __str__).
    """
    text: str = field(init=False)
    fake_probability: Optional[float] = None   # in [0,1]
    label: Optional[str] = None                # "Fake" / "Real" (based on threshold)
    error: Optional[str] = None                # optional diagnostics

    def is_useful(self) -> Optional[bool]:
        """
        Returns True iff a usable probability is available.
        Planner behavior:
        - If False/None: do not base decisions on this tool; consider other tools instead.
        """
        return self.fake_probability is not None

    def __str__(self):
        if self.fake_probability is None:
            return "Deepfake Detection Results\nScore: N/A"
        prob = self.fake_probability
        label = self.label or ("Fake" if prob >= 0.5 else "Real")
        # Replace the Unicode arrow with ASCII
        return f"Deepfake Detection Results\nFake probability: {prob:.3f} -> {label}"

    def __post_init__(self):
        # Auto-generate the summary text upon initialization.
        self.text = str(self)


class DeepfakeDetector(Tool):
    """
    TOOL WRAPPER around Effort/DeepfakeBench.

    FUNCTION:
    ---------
    • Converts DetectDeepfake actions into a model inference call.
    • Returns DeepfakeResults with a calibrated per-image probability in [0, 1].

    FAILURE POLICY:
    ---------------
    • Never raises from `_perform`. On any error, returns non-useful result with `error` set.
    • If the optional landmark model is missing/unreadable, it proceeds without alignment.

    WHEN THE LLM SHOULD PICK THIS TOOL:
    -----------------------------------
    • Whenever the current claim includes an image and a *deepfake likelihood* is relevant
      (e.g., verifying authenticity, detecting face swaps, etc.).
    """
    name = "deepfake_detector"
    actions = [DetectDeepfake]
    summarize = False  # This tool already returns concise, ready-to-display text via DeepfakeResults.text.

    def __init__(
        self,
        detector_config: str = "third_party/effort/DeepfakeBench/training/effort_config/detector/effort.yaml",
        weights: str = "third_party/effort/DeepfakeBench/training/weights/effort_clip_L14_trainOn_FaceForensic.pth",
        landmark_model: Optional[str] = "third_party/effort/DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat",
        threshold: float = 0.5,
        **kwargs
    ):
        """
        CONSTRUCTOR (LLM-ORIENTED PARAMETER GUIDE)
        ------------------------------------------
        detector_config : str
            Path to the Effort/DeepfakeBench detector YAML config.
            • Must match the architecture expected by the provided `weights`.
            • If not found, the tool returns non-useful (fake_probability=None).

        weights : str
            Path to the detector's pretrained weights (.pth).
            • Should be compatible with `detector_config`.
            • If not found, the tool returns non-useful.

        landmark_model : Optional[str]
            Path to the optional dlib 81-landmarks .dat file for face alignment.
            • If provided and exists → enables alignment for potentially better accuracy.
            • If missing/unreadable → tool logs a warning and proceeds WITHOUT alignment.
            • If you do not want alignment, pass None.

        threshold : float
            Decision boundary used to translate probability → label:
              label = "Fake" if prob >= threshold else "Real"
            • Default 0.5. Adjust if your application prefers a more conservative boundary.

        **kwargs :
            Extra keyword args for the base Tool (e.g., device, logging configs, etc.).

        DEVICE POLICY:
        --------------
        • Respects DEFAME's injected device if set; else defaults to CUDA if available.
        """
        super().__init__(**kwargs)
        self.detector_config = detector_config
        self.weights = weights
        self.landmark_model = landmark_model
        self.threshold = float(threshold)
        # Keep DEFAME device semantics: use pre-set device if provided, else prefer CUDA.
        self.device = torch.device(self.device if self.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    def _perform(self, action: DetectDeepfake) -> DeepfakeResults:
        """
        CORE EXECUTION: run Effort on the provided image and return a robust result.

        LLM EXPECTATIONS:
        ------------------
        • Input: DetectDeepfake(action) with a valid image reference (local path/URL supported by ezmm).
        • Output: DeepfakeResults with `fake_probability` in [0, 1], or non-useful if unavailable.
        """
        try:
            # 1) Validate required assets exist (hard requirements).
            for pth, desc in [
                (self.detector_config, "detector_config"),
                (self.weights, "weights"),
            ]:
                if not Path(pth).exists():
                    msg = f"{desc} not found at {pth}"
                    logger.warning(f"[deepfake_detector] {msg}")
                    return DeepfakeResults(fake_probability=None, label=None, error=msg)

            # 2) Landmark model is optional (soft requirement).
            #    If not found, run without alignment.
            if self.landmark_model and not Path(self.landmark_model).exists():
                logger.warning(f"[deepfake_detector] landmark_model not found at {self.landmark_model}; proceeding without face alignment.")
                landmark_model = None
            else:
                landmark_model = self.landmark_model

            # 3) Resolve the image from EZMM wrapper into a PIL image.
            pil_img: PILImage = action.image.image
            if pil_img is None:
                msg = "No image available on action.image"
                logger.warning(f"[deepfake_detector] {msg}")
                return DeepfakeResults(fake_probability=None, label=None, error=msg)

            # 4) Run the Effort detector (returns a float probability for class=FAKE).
            prob = effort_fake_probability(
                image_pil=pil_img,
                detector_config=self.detector_config,
                weights=self.weights,
                landmark_model=landmark_model,
            )

            # 5) Handle upstream 'no score' condition (e.g., no face & skip policy).
            if prob is None:
                logger.info("[deepfake_detector] detector returned None (e.g., no face) → skipping.")
                return DeepfakeResults(fake_probability=None, label=None, error="no_face_or_unavailable")

            # 6) Map probability to a binary label using the configured threshold.
            label = "Fake" if prob >= self.threshold else "Real"
            return DeepfakeResults(fake_probability=float(prob), label=label)

        except Exception as e:
            # Defensive: never break the pipeline. Return non-useful with diagnostics.
            logger.warning(f"[deepfake_detector] failed with error: {e}")
            return DeepfakeResults(fake_probability=None, label=None, error=str(e))

    def _summarize(self, result: DeepfakeResults, **kwargs) -> Optional[MultimodalSequence]:
        """
        OPTIONAL: Tool-level summarization for UI/logs.

        • If non-useful: display a friendly message (do not throw).
        • If useful: present the probability and decision label.
        """
        if result.fake_probability is None:
            return MultimodalSequence("Deepfake detector unavailable or not applicable.")
        return MultimodalSequence(f"Deepfake Detector fake probability: {result.fake_probability:.3f} → {result.label}")
