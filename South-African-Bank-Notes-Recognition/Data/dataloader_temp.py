"""
dataloader_temp.py — Temporary dataloader for ResNet-18 training.

OUTPUT PER SAMPLE:
    image_tensor  — torch.Tensor (3, 224, 224) float32 [0, 1]
    label         — int  0=R10  1=R20  2=R50  3=R100  4=R200
"""

import os, re, random, sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Data.preprocessing import apply_gaussian_smoothing, equalize_clahe

DENOMINATION_TO_LABEL = {"R10":0, "R20":1, "R50":2, "R100":3, "R200":4}
LABEL_TO_DENOMINATION = {v: k for k, v in DENOMINATION_TO_LABEL.items()}
FILENAME_PATTERN      = re.compile(
    r'^(R\d{2,3})_(Front|Back)_\d{1,3}_\d{4}\.png$', re.IGNORECASE
)
IMAGE_SIZE = (224, 224)


class BanknoteDataset(Dataset):
    """
    Dataset with full augmentation pipeline:
        Load → Preprocessing (CLAHE) → Colour jitter → Background composite
        → Perspective warp → Rotation/scale → Gaussian noise → Resize → Tensor
    """

    def __init__(self, root: str, augment: bool = False, colour: bool = True):
        self.root    = root
        self.augment = augment
        self.colour  = colour
        self.samples = []
        self._scan_folder()
        if not self.samples:
            raise RuntimeError(f"No valid images found in '{root}'.")
        print(f"BanknoteDataset: found {len(self.samples)} images in '{root}'")
        self._print_class_counts()

    def _scan_folder(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset folder not found: '{self.root}'")
        for fn in sorted(os.listdir(self.root)):
            m = FILENAME_PATTERN.match(fn)
            if not m:
                continue
            denom = m.group(1).upper()
            if denom not in DENOMINATION_TO_LABEL:
                continue
            self.samples.append((os.path.join(self.root, fn),
                                  DENOMINATION_TO_LABEL[denom]))

    def _print_class_counts(self):
        counts = {k: 0 for k in DENOMINATION_TO_LABEL}
        for _, l in self.samples:
            counts[LABEL_TO_DENOMINATION[l]] += 1
        print("  Class distribution:")
        for d, c in counts.items():
            print(f"    {d}: {c} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]

        # --- Load ---------------------------------------------------------
        bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not load: {filepath}")

        # --- Preprocessing (CLAHE on V channel) ---------------------------
        bgr = self._apply_preprocessing(bgr)

        # --- Colour jitter ------------------------------------------------
        # Applied BEFORE background compositing so jitter affects the note.
        # This is the most impactful augmentation for the R50/R100 confusion:
        # training under varied hues forces the model to learn shape/texture
        # features rather than relying purely on colour.
        if self.augment:
            bgr = self._colour_jitter(bgr)

        # --- Background compositing (70% of augmented samples) -----------
        if self.augment and self.colour and random.random() < 0.7:
            rgb_tmp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_tmp = self._apply_background(rgb_tmp)
            bgr     = cv2.cvtColor(rgb_tmp, cv2.COLOR_RGB2BGR)

        # --- Convert to output colour space ------------------------------
        if self.colour:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # --- Perspective warp (50% of augmented samples) ----------------
        if self.augment and random.random() < 0.5:
            image = self._perspective_warp(image)

        # --- Rotation + scale augmentation --------------------------------
        if self.augment:
            image = self._augment_rotation_scale(image)

        # --- Gaussian noise (40% of augmented samples) -------------------
        if self.augment and random.random() < 0.4:
            image = self._add_gaussian_noise(image)

        # --- Resize → tensor [0, 1] --------------------------------------
        image        = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        image        = image.astype(np.float32) / 255.0

        if self.colour:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0)

        return image_tensor, label

    # ------------------------------------------------------------------
    def _apply_preprocessing(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = apply_gaussian_smoothing(v, kernel_size=3)
        v = equalize_clahe(v, clip_limit=2.0, grid_size=(8, 8))
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    # ------------------------------------------------------------------
    def _colour_jitter(self, bgr: np.ndarray) -> np.ndarray:
        """
        Randomly adjusts brightness and contrast only.

        Hue and saturation jitter removed — with only 68 training images,
        colour is the most reliable discriminating feature between
        denominations (R10=green, R20=brown, R50=purple, R100=blue,
        R200=orange). Shifting hue destroys this signal and hurts accuracy.

        Brightness and contrast variation covers the camera vs scanner
        exposure difference without destroying colour information.
        """
        alpha = random.uniform(0.7, 1.3)   # contrast
        beta  = random.randint(-30, 30)    # brightness
        return cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)

    # ------------------------------------------------------------------
    def _perspective_warp(self, image: np.ndarray) -> np.ndarray:
        """
        Random perspective transform simulating camera angle variation.
        Moves each corner by up to 8% of the image dimension.
        """
        h, w  = image.shape[:2]
        margin = int(min(h, w) * 0.08)
        src   = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst   = np.float32([
            [random.randint(0, margin),        random.randint(0, margin)],
            [random.randint(w-margin, w),      random.randint(0, margin)],
            [random.randint(w-margin, w),      random.randint(h-margin, h)],
            [random.randint(0, margin),        random.randint(h-margin, h)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (w, h))

    # ------------------------------------------------------------------
    def _augment_rotation_scale(self, image: np.ndarray) -> np.ndarray:
        """Random rotation (0–359°) and scale (0.5–1.5×)."""
        rotation = random.randint(0, 359)
        scale    = round(random.uniform(0.5, 1.5), 2)
        h, w     = image.shape[:2]
        center   = (w // 2, h // 2)
        M        = cv2.getRotationMatrix2D(center, rotation, scale)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w    = int(h * sin + w * cos)
        new_h    = int(h * cos + w * sin)
        M[0, 2] += new_w / 2 - center[0]
        M[1, 2] += new_h / 2 - center[1]
        return cv2.warpAffine(image, M, (new_w, new_h))

    # ------------------------------------------------------------------
    def _add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Adds random Gaussian noise to simulate sensor/compression noise."""
        sigma = random.uniform(5, 20)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def _apply_background(self, rgb: np.ndarray) -> np.ndarray:
        """Pastes note onto a synthetic background (white pixel masking)."""
        h, w    = rgb.shape[:2]
        bg      = self._generate_background(h, w)
        gray    = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        kernel  = np.ones((3, 3), np.uint8)
        mask    = cv2.erode(mask, kernel, iterations=1)
        mask3   = cv2.merge([mask, mask, mask])
        return np.where(mask3 == 255, bg, rgb).astype(np.uint8)

    def _generate_background(self, h: int, w: int) -> np.ndarray:
        """Generates one of four synthetic background types."""
        t = random.randint(0, 3)
        if t == 0:
            return np.full((h, w, 3),
                           [random.randint(30, 220) for _ in range(3)],
                           dtype=np.uint8)
        elif t == 1:
            c1  = np.array([random.randint(30, 220) for _ in range(3)], np.float32)
            c2  = np.array([random.randint(30, 220) for _ in range(3)], np.float32)
            a   = np.linspace(0, 1, w, dtype=np.float32)
            row = c1[np.newaxis, :] * (1 - a[:, np.newaxis]) + \
                  c2[np.newaxis, :] * a[:, np.newaxis]
            return np.tile(row[np.newaxis], (h, 1, 1)).astype(np.uint8)
        elif t == 2:
            base  = [random.randint(50, 180) for _ in range(3)]
            noise = np.random.randint(-40, 40, (h, w, 3), dtype=np.int16)
            return np.clip(np.array(base, np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            bg  = np.full((h, w, 3),
                          [random.randint(100, 200) for _ in range(3)],
                          dtype=np.uint8)
            lc  = [random.randint(30, 100) for _ in range(3)]
            sp  = random.randint(15, 40)
            for y in range(0, h, sp): bg[y, :] = lc
            for x in range(0, w, sp): bg[:, x] = lc
            return bg


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    root = os.path.join(_PROJECT_ROOT, "Dataset", "raw",
                        "Banknote_Dataset_(2005-2023)")
    ds   = BanknoteDataset(root=root, augment=True, colour=True)
    img, lbl = ds[0]
    print(f"Shape: {img.shape}  Label: {LABEL_TO_DENOMINATION[lbl]}")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    imgs, lbls = next(iter(loader))
    print(f"Batch: {imgs.shape}  Labels: {[LABEL_TO_DENOMINATION[l.item()] for l in lbls]}")