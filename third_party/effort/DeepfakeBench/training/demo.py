import numpy as np
import cv2
import yaml
from PIL import Image as pil_image
import dlib
import torch
import torch.nn.functional as F  # (kept if your DETECTOR model uses it)
from torchvision import transforms
from third_party.effort.DeepfakeBench.training.trainer.trainer import Trainer  # if unused, you can remove
from detectors import DETECTOR
from imutils import face_utils
from skimage import transform as trans
import torchvision.transforms as T
from pathlib import Path
from typing import Tuple, List
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def inference(model, data_dict):
    data, label = data_dict['image'], data_dict['label']
    # move data to GPU
    data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
    predictions = model(data_dict, inference=True)
    return predictions


def get_keypts(image, face, predictor, face_detector):
    """
    Extract 5 keypoints (left eye, right eye, nose tip, left mouth, right mouth)
    using an 81-point dlib predictor.
    """
    shape = predictor(image, face)

    leye   = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye   = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose   = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)

    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=224, mask=None):
    """
    Try to detect a face, align it to 5 keypoints, and return:
      (cropped_face_bgr, landmark_81_np_or_None, original_dlib_face_rect_or_None)

    This function ALWAYS returns 3 values to avoid unpack errors.
    If detection/alignment fails, it returns (image, None, None) so callers can fall back.
    """
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        # align and crop the face according to the given 5-point landmarks
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]
        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.0
        y_margin = target_size[1] * margin_rate / 2.0

        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        warped = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            warped = cv2.resize(warped, (outsize[1], outsize[0]))

        if mask is not None:
            mask_w = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask_w = cv2.resize(mask_w, (outsize[1], outsize[0]))
            return warped, mask_w
        else:
            return warped

    # Convert to RGB for dlib
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_detector(rgb, 1) if face_detector is not None else []
    if len(faces) == 0:
        # No face: return original image so downstream can proceed on full frame
        return image, None, None

    # Take the largest face
    face = max(faces, key=lambda rect: rect.width() * rect.height())

    # 5-point landmarks for alignment
    try:
        landmarks_5 = get_keypts(rgb, face, predictor, face_detector)
    except Exception:
        # If predictor fails, fall back
        return image, None, face

    # Align & crop to desired resolution
    try:
        cropped_rgb = img_align_crop(rgb, landmarks_5, outsize=(res, res), mask=mask)
        if isinstance(cropped_rgb, tuple):
            cropped_rgb = cropped_rgb[0]  # (img, mask) -> img
        cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        # Warping failed -> fall back to original image
        return image, None, face

    # Optionally extract 81 landmarks on the aligned face (safe-guard if not found)
    try:
        faces_aligned = face_detector(cropped_bgr[:, :, ::-1], 1)  # dlib expects RGB
        if len(faces_aligned) > 0:
            lm = predictor(cropped_bgr[:, :, ::-1], faces_aligned[0])
            lm_np = face_utils.shape_to_np(lm)
        else:
            lm_np = None
    except Exception:
        lm_np = None

    return cropped_bgr, lm_np, face


def load_detector(detector_cfg: str, weights: str):
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)  # keep strict=False if heads differ
    model.eval()
    print("[✓] Detector loaded.")
    return model


def preprocess_face(img_bgr: np.ndarray):
    """BGR → normalized tensor (1×3×224×224)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]),
    ])
    return transform(pil_image.fromarray(img_rgb)).unsqueeze(0)


@torch.inference_mode()
def infer_single_image(
    img_bgr: np.ndarray,
    face_detector,
    landmark_predictor,
    model,
) -> Tuple[int, float]:
    """
    Run detection; returns (cls_out, prob).
    Falls back to full image if no face/alignment is available.
    """
    face_aligned = img_bgr
    if face_detector is not None and landmark_predictor is not None:
        fa, _, _ = extract_aligned_face_dlib(face_detector, landmark_predictor, img_bgr, res=224)
        if isinstance(fa, np.ndarray) and fa.ndim == 3:
            face_aligned = fa  # use aligned face
        else:
            face_aligned = img_bgr  # fallback

    face_tensor = preprocess_face(face_aligned).to(device)
    data = {"image": face_tensor, "label": torch.tensor([0]).to(device)}
    preds = inference(model, data)
    cls_out = preds["cls"].squeeze().detach().cpu().numpy()
    prob = preds["prob"].squeeze().detach().cpu().numpy()
    return cls_out, prob


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def collect_image_paths(path_str: str) -> List[Path]:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"[Error] Path does not exist: {path_str}")

    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"[Error] Invalid image format: {p.name}")
        return [p]

    img_list = [fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() in IMG_EXTS]
    if not img_list:
        raise RuntimeError(f"[Error] No valid image files found in directory: {path_str}")

    return sorted(img_list)


def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake image inference (single image version)"
    )
    p.add_argument("--detector_config", default='training/effort_config/detector/effort.yaml',
                   help="Path to YAML config")
    p.add_argument("--weights", required=True, help="Path to detector weights")
    p.add_argument("--image", required=True, help="Image file or directory")
    p.add_argument("--landmark_model", default=False,
                   help="Path to dlib 81-landmarks .dat (or False to disable)")
    return p.parse_args()


def main():
    args = parse_args()

    model = load_detector(args.detector_config, args.weights)
    if args.landmark_model:
        face_det = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(args.landmark_model)
    else:
        face_det, shape_predictor = None, None

    img_paths = collect_image_paths(args.image)
    if len(img_paths) > 1:
        print(f"Collected {len(img_paths)} images in total; inferring...\n")

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Warning] failed to load, skipping: {img_path}")
            continue

        cls, prob = infer_single_image(img, face_det, shape_predictor, model)
        # Ensure scalars for printing
        cls_scalar = int(np.ravel(cls)[-1])
        prob_scalar = float(np.ravel(prob)[-1])
        print(
            f"[{idx}/{len(img_paths)}] {img_path.name:>30} | Pred Label: {cls_scalar} "
            f"(0=Real, 1=Fake) | Fake Prob: {prob_scalar:.4f}"
        )


# -------- Public API with simple caching (for programmatic use) --------

_EFFORT_CACHE = {}

def _get_cached(key):
    return _EFFORT_CACHE.get(key)

def _set_cached(key, value):
    _EFFORT_CACHE[key] = value
    return value

def effort_fake_probability(
    image_pil: pil_image,
    detector_config: str,
    weights: str,
    landmark_model: str | None = None,
) -> float:
    """
    Compute deepfake probability for a single PIL image.
    Returns probability of class 1 (fake) in [0,1].
    """
    cache_key = (
        Path(detector_config).resolve(),
        Path(weights).resolve(),
        Path(landmark_model).resolve() if landmark_model else None
    )
    cached = _get_cached(cache_key)

    if cached is None:
        model = load_detector(str(cache_key[0]), str(cache_key[1]))
        if landmark_model:
            face_det = dlib.get_frontal_face_detector()
            shape_predictor = dlib.shape_predictor(str(cache_key[2]))
        else:
            face_det, shape_predictor = None, None

        cached = _set_cached(cache_key, {
            "model": model,
            "face_det": face_det,
            "shape_predictor": shape_predictor,
        })

    model = cached["model"]
    face_det = cached["face_det"]
    shape_predictor = cached["shape_predictor"]

    image_rgb = image_pil.convert("RGB")
    img_np = np.array(image_rgb)  # HxWx3 RGB
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    cls, prob = infer_single_image(
        img_bgr=img_bgr,
        face_detector=face_det,
        landmark_predictor=shape_predictor,
        model=model,
    )

    if isinstance(prob, np.ndarray):
        prob = float(np.ravel(prob)[-1])
    else:
        prob = float(prob)

    return prob


if __name__ == "__main__":
    main()
