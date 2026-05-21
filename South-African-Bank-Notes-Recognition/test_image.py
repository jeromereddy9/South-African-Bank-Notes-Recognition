"""
test_image.py — Single-image inference for the trained ResNet-18 model.

Usage:
    python test_image.py <path_to_image>              # clean scan
    python test_image.py <path_to_image> --segment    # real-world photo
"""

import os, sys, importlib.util, argparse
import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load ResNet-18 (handles hyphen in filename)
_spec = importlib.util.spec_from_file_location(
    "ResNet18", os.path.join(PROJECT_ROOT, "Models", "ResNet-18.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_model   = _mod.load_model
CLASS_LABELS = _mod.CLASS_LABELS
NUM_CLASSES  = _mod.NUM_CLASSES

from Data.preprocessing import apply_gaussian_smoothing, equalize_clahe
from Data.segmentation  import segment_note

CHECKPOINT    = os.path.join(PROJECT_ROOT, "checkpoints", "resnet18_best.pth")
IMAGE_SIZE    = (224, 224)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_and_normalise(image_path: str,
                             apply_segmentation: bool = False) -> torch.Tensor:
    """
    Preprocessing pipeline:
        1. Load BGR
        2. (Optional) GrabCut segmentation — only for real-world photos
        3. CLAHE on V channel — only for clean scans (not real-world photos)
        4. RGB, resize 224x224, float [0,1]
        5. ImageNet normalisation
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Could not load image: {image_path}")

    # Step 2 — Segmentation only for real-world photos
    if apply_segmentation:
        bgr = segment_note(bgr, method='canny')

    # Step 3 — CLAHE only for clean scans
    # Real-world photos already have natural contrast — CLAHE amplifies
    # camera-specific tones and widens the domain gap
    if not apply_segmentation:
        hsv     = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v       = apply_gaussian_smoothing(v, kernel_size=3)
        v       = equalize_clahe(v, clip_limit=2.0, grid_size=(8, 8))
        bgr     = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    # Step 4-5
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb    = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def predict_with_tta(model, image_path: str, device: torch.device,
                     apply_segmentation: bool = False) -> dict:
    """TTA inference — averages 5 augmented versions for robust prediction."""
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if apply_segmentation:
        bgr = segment_note(bgr, method='canny')

    # CLAHE only for clean scans
    if not apply_segmentation:
        hsv     = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v       = apply_gaussian_smoothing(v, kernel_size=3)
        v       = equalize_clahe(v, clip_limit=2.0, grid_size=(8, 8))
        bgr     = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb     = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

    def to_tensor(img):
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return (t - IMAGENET_MEAN) / IMAGENET_STD

    variants = [
        rgb,
        cv2.flip(rgb, 1),
        cv2.convertScaleAbs(rgb, alpha=1.0, beta=20),
        cv2.convertScaleAbs(rgb, alpha=1.0, beta=-20),
        cv2.warpAffine(rgb,
            cv2.getRotationMatrix2D((112, 112), 10, 1.0), IMAGE_SIZE),
    ]

    model.eval()
    avg_probs = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for v_img in variants:
            t         = to_tensor(v_img).unsqueeze(0).to(device)
            logits    = model(t)
            probs     = torch.softmax(logits, dim=1).squeeze(0).cpu()
            avg_probs += probs

    avg_probs /= len(variants)
    confidence, predicted_idx = torch.max(avg_probs, dim=0)

    return {
        "label"        : CLASS_LABELS[predicted_idx.item()],
        "class_index"  : predicted_idx.item(),
        "confidence"   : round(confidence.item(), 4),
        "probabilities": {
            CLASS_LABELS[i]: round(avg_probs[i].item(), 4)
            for i in range(NUM_CLASSES)
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to banknote image")
    parser.add_argument("--segment", action="store_true",
                        help="Apply GrabCut segmentation (use for real-world photos)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model  : {CHECKPOINT}")
    print(f"Testing image  : {args.image_path}")
    print(f"Segmentation   : {'ON (real-world photo)' if args.segment else 'OFF (clean scan)'}")
    print(f"Device         : {device}\n")

    model  = load_model(CHECKPOINT, device)
    tensor = preprocess_and_normalise(args.image_path, args.segment).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx    = probs.argmax().item()
    single_pred = CLASS_LABELS[pred_idx]
    single_conf = probs[pred_idx].item()

    tta_result = predict_with_tta(model, args.image_path, device, args.segment)

    print("=" * 45)
    print(f"  Single prediction : {single_pred}  ({single_conf*100:.1f}%)")
    print(f"  TTA prediction    : {tta_result['label']}  ({tta_result['confidence']*100:.1f}%)")
    print("=" * 45)
    print("\nProbabilities (TTA averaged):")
    for denom, prob in sorted(tta_result['probabilities'].items(),
                               key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"  {denom:>4}  {prob*100:5.1f}%  {bar}")