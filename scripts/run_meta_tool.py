from pathlib import Path
from dataclasses import is_dataclass, asdict
import json, sys

from defame.evidence_retrieval.tools import MetadataAuthVerifier, VerifyImageAuthenticity
# Import your tool + action (UNCHANGED API import path)


# --- optional: local-file action creator (bypasses ezmm if needed) ---
class _LocalImageRef:
    def __init__(self, file_path: str, reference: str = "<local:image>"):
        self.file_path = file_path
        self.reference = reference

def make_local_action(img_path: str) -> VerifyImageAuthenticity:
    # bypass __init__ to directly set an ezmm-compatible image-like object
    a = VerifyImageAuthenticity.__new__(VerifyImageAuthenticity)
    a.image = _LocalImageRef(file_path=img_path)
    return a

def sanitize(obj):
    """Recursively convert to JSON-serializable structures, stringifying the rest."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return sanitize(asdict(obj))
    # Fallback: string representation
    return str(obj)

# ---- INPUT ----
IMAGE = Path("D:/datasets/flickr_metadata/imgs/4 February 2012 - Men and Women flee tear gas.jpg").resolve()
if not IMAGE.exists():
    print(f"[error] File not found: {IMAGE}", file=sys.stderr)
    sys.exit(1)

# ---- RUN TOOL ----
tool = MetadataAuthVerifier(model="gpt-4o", max_meta_len=2000, config_path="config/api_keys.yaml")

# Prefer standard action; if ezmm complains, switch to make_local_action(str(IMAGE))
try:
    action = VerifyImageAuthenticity(image=str(IMAGE))
except Exception:
    action = make_local_action(str(IMAGE))

res = tool.perform(action)

# --- The tool's public __str__ usually mirrors the primary text; still show it for quick glance ---
print("\n=== Pretty Output ===")
print(res)

