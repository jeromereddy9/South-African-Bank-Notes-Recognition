import os, re, random, sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# We keep this for the CLAHE enhancement applied to the already segmented images
from Data.preprocessing import preprocessing_CLAHE

DENOMINATION_TO_LABEL = {"R10":0, "R20":1, "R50":2, "R100":3, "R200":4}
LABEL_TO_DENOMINATION = {v: k for k, v in DENOMINATION_TO_LABEL.items()}
FILENAME_PATTERN      = re.compile(
    r'^(R\d{2,3})_(Front|Back)_\d{1,3}_\d{4}\.png$', re.IGNORECASE
)
IMAGE_SIZE = (224, 224)


class BanknoteDataset(Dataset):

    def __init__(self, root: str, augment: bool = False, colour: bool = True):
        self.root    = root
        self.augment = augment
        self.colour  = colour
        self.samples = []
        self._scan_folder()
        if not self.samples:
            raise RuntimeError(f"No valid images found in '{root}'.")
        print(f"BanknoteDataset: loaded {len(self.samples)} pre-segmented images from '{root}'")
        self._print_class_counts()

    def _scan_folder(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset folder not found: '{self.root}'")
        for fn in sorted(os.listdir(self.root)):
            m = FILENAME_PATTERN.match(fn)
            if not m: continue
            denom = m.group(1).upper()
            if denom not in DENOMINATION_TO_LABEL: continue
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

        # 1. DIRECT LOAD: Images are already segmented offline.
        bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not load pre-segmented image: {filepath}")

        # 2. COLOR-PRESERVING ENHANCEMENT: Apply CLAHE to the pre-segmented image.
        bgr = preprocessing_CLAHE(bgr)

        # 3. AUGMENTATIONS (Training only)
        if self.augment:
            bgr = self._colour_jitter(bgr)
            if random.random() < 0.2:
                bgr = self._strong_augmentation(bgr)

        # 4. BACKGROUND COMPOSITING 
        if self.augment and self.colour and random.random() < 0.7:
            rgb_tmp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_tmp = self._apply_background(rgb_tmp)
            bgr     = cv2.cvtColor(rgb_tmp, cv2.COLOR_RGB2BGR)

        # 5. CONVERSION
        if self.colour:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 6. GEOMETRIC AUGMENTATIONS
        if self.augment and random.random() < 0.5:
            image = self._perspective_warp(image)
        if self.augment:
            image = self._augment_rotation_scale(image)
        if self.augment and random.random() < 0.4:
            image = self._add_gaussian_noise(image)

        # 7. RESIZE & NORMALIZE
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0

        if self.colour:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0)

        return image_tensor, label

    # ... (Keep all your existing helper methods below this point) ...
    def _colour_jitter(self, bgr: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.7, 1.3)   
        beta  = random.randint(-30, 30)    
        return cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)

    def _strong_augmentation(self, bgr: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.5, 1.5)
        beta = random.randint(-50, 50)
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
        if random.random() < 0.5:
            kernel = random.choice([3, 5])
            bgr = cv2.GaussianBlur(bgr, (kernel, kernel), 0)
        return bgr

    def _perspective_warp(self, image: np.ndarray) -> np.ndarray:
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

    def _augment_rotation_scale(self, image: np.ndarray) -> np.ndarray:
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

    def _add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        sigma = random.uniform(5, 20)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _apply_background(self, rgb: np.ndarray) -> np.ndarray:
        h, w    = rgb.shape[:2]
        bg      = self._generate_background(h, w)
        gray    = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        kernel  = np.ones((3, 3), np.uint8)
        mask    = cv2.erode(mask, kernel, iterations=1)
        mask3   = cv2.merge([mask, mask, mask])
        return np.where(mask3 == 255, bg, rgb).astype(np.uint8)

    def _generate_background(self, h: int, w: int) -> np.ndarray:
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