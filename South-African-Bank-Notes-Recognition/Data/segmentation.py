import cv2
import numpy as np
from Data.preprocessing import (
    apply_gaussian_smoothing,
    preprocessing_CLAHE
)

def detect_edges_canny(image, low_threshold=40, high_threshold=120):

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Smooth to reduce noise
    smoothed = apply_gaussian_smoothing(image, kernel_size=5)
    # Apply Canny edge detection
    return cv2.Canny(smoothed, low_threshold, high_threshold)


def detect_edges_sobel(image):

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Smooth to reduce noise
    smoothed = apply_gaussian_smoothing(image, kernel_size=5)
    # Compute gradients in X and Y directions
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate gradient magnitude
    magnitude = np.abs(gx) + np.abs(gy)
    # Normalize to 0-255 range
    edge_map = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply threshold to get binary edge map
    _, edge_map = cv2.threshold(edge_map, 40, 255, cv2.THRESH_BINARY)
    return edge_map


def segment_note(image, use_clahe=False):

    if image is None or image.size == 0:
        return None
    
    # Ensure input is 3-channel BGR for consistent processing
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return _grabcut_and_warp(image)


def _grabcut_and_warp(image):
 
    OUT_W, OUT_H = 640, 300  # Target output dimensions
    h, w = image.shape[:2]
    
    # Scale down for faster GrabCut processing 
    MAX_W = 800
    scale = min(1.0, MAX_W / w)
    work = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image.copy()
    
    # Initialize GrabCut models
    mask = np.zeros(work.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define bounding box 
    margin = 0.05
    rect = (int(work.shape[1] * margin),
            int(work.shape[0] * margin),
            int(work.shape[1] * (1 - 2 * margin)),
            int(work.shape[0] * (1 - 2 * margin)))
    
    try:
        # Run GrabCut to separate foreground from background
        cv2.grabCut(work, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        # Create foreground mask 
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        # Fallback: return resized original if GrabCut fails
        return cv2.resize(image, (OUT_W, OUT_H))
    
    # Find contours of the foreground
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (OUT_W, OUT_H))
    
    # Get the largest contour 
    largest = max(contours, key=cv2.contourArea)
    
    # Get minimum area rectangle 
    min_rect = cv2.minAreaRect(largest)
    corners = cv2.boxPoints(min_rect)
    corners = np.int32(corners)
    
    # Apply perspective transform to straighten the note
    note = _perspective_transform(image, corners, scale, OUT_W, OUT_H)
    return note


def _perspective_transform(image, corners, scale, out_w, out_h):

    # Convert corners to float and rescale to original image size
    pts = corners.reshape(-1, 2).astype(np.float32)
    if scale < 1.0:
        pts = pts / scale  # Project corners back to full original size coordinates
    
    # Sort corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)      # Sum of coordinates (x + y)
    d = np.diff(pts, axis=1) # Difference of coordinates (y - x)
    
    ordered[0] = pts[np.argmin(s)]  # x + y is minimum = Top-Left
    ordered[2] = pts[np.argmax(s)]  # x + y is maximum = Bottom-Right
    ordered[1] = pts[np.argmin(d)]  # y - x is minimum = Top-Right
    ordered[3] = pts[np.argmax(d)]  # y - x is maximum = Bottom-Left
    
    # Slight internal crop (2%) to remove background edge artifacts
    centroid = ordered.mean(axis=0)
    ordered = ordered + 0.02 * (centroid - ordered)
    
    # Destination points (straight rectangle)
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)
    
    # Compute perspective transform matrix and apply
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    
    return warped