import cv2
import numpy as np
from Data.preprocessing import (
    apply_gaussian_smoothing,
    preprocessing_global,
    preprocessing_CLAHE
)

def detect_edges_canny(image, low_threshold=40, high_threshold=120):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = apply_gaussian_smoothing(image, kernel_size=5)
    return cv2.Canny(smoothed, low_threshold, high_threshold)

def detect_edges_sobel(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = apply_gaussian_smoothing(image, kernel_size=5)
    
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.abs(gx) + np.abs(gy)
    edge_map = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_map = cv2.threshold(edge_map, 40, 255, cv2.THRESH_BINARY)
    return edge_map

def segment_note(image, use_clahe=False):
    """
    Isolates the banknote using a highly robust GrabCut + minAreaRect hybrid pipeline.
    Returns a clean, perspective-warped 3-Channel BGR image.
    """
    if image is None or image.size == 0:
        return None
    
    # Ensure input is 3-channel BGR for consistent processing
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    return _grabcut_and_warp(image)

def _grabcut_and_warp(image):
    OUT_W, OUT_H = 640, 300
    h, w = image.shape[:2]
    
    # Scale down for speed during GrabCut execution
    MAX_W = 800
    scale = min(1.0, MAX_W / w)
    work = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image.copy()
    
    # GrabCut Initialization with a conservative bounding box margin
    mask = np.zeros(work.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    margin = 0.05
    rect = (int(work.shape[1] * margin),
            int(work.shape[0] * margin),
            int(work.shape[1] * (1 - 2 * margin)),
            int(work.shape[0] * (1 - 2 * margin)))
    
    try:
        cv2.grabCut(work, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        # Fallback if GrabCut encounters an execution exception
        return cv2.resize(image, (OUT_W, OUT_H))
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (OUT_W, OUT_H))
    
    largest = max(contours, key=cv2.contourArea)
    
    # GUARANTEE 4 CORNERS: Use Minimum Area Oriented Bounding Box
    min_rect = cv2.minAreaRect(largest)
    corners = cv2.boxPoints(min_rect)
    corners = np.int32(corners)
    
    # Warp perspective on the ORIGINAL color image using upscale tracking
    note = _perspective_transform(image, corners, scale, OUT_W, OUT_H)
    return note

def _perspective_transform(image, corners, scale, out_w, out_h):
    pts = corners.reshape(-1, 2).astype(np.float32)
    if scale < 1.0:
        pts = pts / scale  # Project corners back to full original size coordinates
    
    # Mathematically sort corners: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    
    ordered[0] = pts[np.argmin(s)]  # x + y is minimum
    ordered[2] = pts[np.argmax(s)]  # x + y is maximum
    ordered[1] = pts[np.argmin(d)]  # y - x is minimum
    ordered[3] = pts[np.argmax(d)]  # y - x is maximum
    
    # Slight internal crop (2%) to clip remaining background edge artifacts
    centroid = ordered.mean(axis=0)
    ordered = ordered + 0.02 * (centroid - ordered)
    
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))