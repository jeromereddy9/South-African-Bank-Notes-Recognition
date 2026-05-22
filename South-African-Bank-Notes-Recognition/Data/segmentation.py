import cv2
import numpy as np



# 1. Edge Detection — Canny
def detect_edges_canny(image: np.ndarray,
                       low_threshold: int = 50,
                       high_threshold: int = 150) -> np.ndarray:
    """
    Detects edges using the Canny algorithm (Canny [1986]).

    The Canny detector satisfies three objectives:
        1. Low error rate
        2. Good localisation
        3. Single response per true edge

    Preprocessing uses a bilateral filter to suppress background textures
    (e.g. wood grain) while preserving the sharp note boundary, followed by
    a moderate Gaussian blur for remaining noise.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bilateral filter — reduces background texture (wood grain) while
    # keeping the sharp note boundary intact
    blurred  = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)

    # Moderate Gaussian for remaining noise
    blurred  = cv2.GaussianBlur(blurred, (7, 7), 0)

    edge_map = cv2.Canny(blurred, low_threshold, high_threshold)
    return edge_map


# 2. Edge Detection — Sobel
def detect_edges_sobel(image: np.ndarray) -> np.ndarray:
    """
    Detects edges using Sobel gradient operators.

        gx = (z7+2z8+z9) - (z1+2z2+z3)
        gy = (z3+2z6+z9) - (z1+2z4+z7)

    Magnitude approximated as |gx| + |gy|.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred  = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)
    blurred  = cv2.GaussianBlur(blurred, (7, 7), 0)

    gx        = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy        = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.abs(gx) + np.abs(gy)

    edge_map  = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_map = cv2.threshold(edge_map, 50, 255, cv2.THRESH_BINARY)
    return edge_map



# 3. Find Largest Contour
def find_largest_contour(edge_map: np.ndarray) -> np.ndarray:
    """
    Finds the largest closed contour — expected to be the note boundary.

    Uses dilation to close gaps in the edge map before contour detection,
    then looks for the largest 4-sided polygon (the rectangular note).
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edge_map, kernel, iterations=3)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Try increasingly aggressive approximation to get a 4-sided polygon
    for contour in contours[:5]:
        perimeter = cv2.arcLength(contour, True)
        for epsilon_factor in [0.02, 0.04, 0.06, 0.08, 0.10]:
            approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
            if len(approx) == 4:
                return approx

    return contours[0]



# 4. Crop to Contour
def crop_to_contour(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Fallback bounding-box crop when perspective transform is unavailable."""
    x, y, w, h = cv2.boundingRect(contour)
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    return image[y:y + h, x:x + w]


# 5. Deskew — Perspective Transform
def deskew(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Applies perspective transform to produce a flat rectangular note crop.
    Output is 640×300 matching SA note aspect ratio (~2.14:1).

    Shrinks each corner point inward toward the contour centroid before
    warping — this ensures the transform starts inside the note boundary
    rather than outside it, eliminating background strips.
    """
    OUT_W, OUT_H = 640, 300

    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) != 4:
        return crop_to_contour(image, contour)

    pts_ordered = _order_points(pts)

    # Shrink corners inward by 4% toward the centroid
    # This moves each corner just inside the note boundary
    centroid     = pts_ordered.mean(axis=0)
    shrink       = 0.04
    pts_shrunk   = pts_ordered + shrink * (centroid - pts_ordered)

    dst = np.array([
        [0,         0        ],
        [OUT_W - 1, 0        ],
        [OUT_W - 1, OUT_H - 1],
        [0,         OUT_H - 1],
    ], dtype=np.float32)

    M      = cv2.getPerspectiveTransform(pts_shrunk, dst)
    warped = cv2.warpPerspective(image, M, (OUT_W, OUT_H))

    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points: top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


# 6. Full Pipeline
def segment_note(image: np.ndarray, method: str = 'canny') -> np.ndarray:
    """
    Full segmentation pipeline — isolates the bank note from background.

    Pipeline:
        1. Edge detection (Canny or Sobel) — used for edge map comparison
        2. GrabCut — robustly separates note (foreground) from background
        3. Bounding rectangle crop + perspective correction
        4. Return 640×300 cropped note
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
           if len(image.shape) == 3 else image.copy()

    # Step 1 — Edge detection (kept for report comparison / edge map output)
    if method == 'canny':
        otsu_thresh, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        edge_map = detect_edges_canny(gray,
                                      int(otsu_thresh * 0.5),
                                      int(otsu_thresh * 1.5))
    else:
        edge_map = detect_edges_sobel(gray)

    # Step 2 — GrabCut foreground/background separation
    # Assumes the note occupies the centre of the image and the background
    # (wood, table etc.) surrounds it near the edges.
    note = _grabcut_crop(image)
    return note


def _grabcut_crop(image: np.ndarray) -> np.ndarray:
    """
    Uses GrabCut to isolate the note from its background.
    Resizes large images before GrabCut to prevent memory allocation errors.
    """
    OUT_W, OUT_H = 640, 300
    h, w         = image.shape[:2]

    # Resize to max 1000px wide before GrabCut — prevents bad allocation
    # on high-resolution photos while preserving enough detail
    MAX_W  = 1000
    scale  = min(1.0, MAX_W / w)
    work_w = int(w * scale)
    work_h = int(h * scale)
    work   = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA) \
             if scale < 1.0 else image.copy()

    # GrabCut on the resized image
    mask      = np.zeros((work_h, work_w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin = 0.08
    rect   = (int(work_w * margin),
              int(work_h * margin),
              int(work_w * (1 - 2 * margin)),
              int(work_h * (1 - 2 * margin)))

    cv2.grabCut(work, mask, rect, bgd_model, fgd_model,
                5, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[segmentation] GrabCut found no foreground — returning resized original.")
        return cv2.resize(image, (OUT_W, OUT_H))

    largest      = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # Scale bounding box back to original image coordinates
    if scale < 1.0:
        x  = int(x  / scale)
        y  = int(y  / scale)
        cw = int(cw / scale)
        ch = int(ch / scale)

    # Small inward margin to remove background fringe
    pad = 8
    x   = max(0, x + pad)
    y   = max(0, y + pad)
    cw  = min(w - x, cw - 2 * pad)
    ch  = min(h - y, ch - 2 * pad)

    cropped = image[y:y + ch, x:x + cw]

    if cropped.size == 0:
        return cv2.resize(image, (OUT_W, OUT_H))

    return cv2.resize(cropped, (OUT_W, OUT_H))


# Quick test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python segmentation.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    img      = cv2.imread(img_path)
    if img is None:
        print(f"Could not load: {img_path}")
        sys.exit(1)

    print(f"Input image size : {img.shape[1]}x{img.shape[0]}")

    for method in ['canny', 'sobel']:
        result   = segment_note(img, method=method)
        out_path = f"segmented_{method}.png"
        cv2.imwrite(out_path, result)
        print(f"[{method}] saved -> {out_path}  ({result.shape[1]}x{result.shape[0]})")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("edges_canny.png", detect_edges_canny(gray))
    cv2.imwrite("edges_sobel.png", detect_edges_sobel(gray))
    print("Edge maps saved.")