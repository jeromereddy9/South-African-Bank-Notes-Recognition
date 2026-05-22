import cv2
import numpy as np
from Data.preprocessing import (
    load_image,
    apply_gaussian_smoothing,
    equalize_histogram,
    equalize_clahe,
    preprocessing_global,
    preprocessing_CLAHE
)


# Edge Detection
def detect_edges_canny(image, low_threshold=50, high_threshold=150):
  
    # Use preprocessing instead of internal blurring
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply smoothing from preprocessing
    smoothed = apply_gaussian_smoothing(image, kernel_size=5)
    
    return cv2.Canny(smoothed, low_threshold, high_threshold)


def detect_edges_sobel(image):
   
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply smoothing from preprocessing
    smoothed = apply_gaussian_smoothing(image, kernel_size=5)
    
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.abs(gx) + np.abs(gy)
    edge_map = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_map = cv2.threshold(edge_map, 50, 255, cv2.THRESH_BINARY)
    return edge_map

#Segmentation
def segment_note(image, use_clahe=False):
    
    if image is None or image.size == 0:
        return None
    
    # Apply preprocessing first
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if use_clahe:
        gray = preprocessing_CLAHE(gray)
    else:
        gray = preprocessing_global(gray)
    
    # Generate edge maps for report
    canny_edges = detect_edges_canny(gray)
    sobel_edges = detect_edges_sobel(gray)
    
    # GrabCut + perspective transform 
    note = _grabcut_and_warp(image)
    
    return note


def _grabcut_and_warp(image):
    OUT_W, OUT_H = 640, 300
    h, w = image.shape[:2]
    
    # Resize if too large
    MAX_W = 1000
    scale = min(1.0, MAX_W / w)
    work = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image.copy()
    
    # GrabCut
    mask = np.zeros(work.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    margin = 0.08
    rect = (int(work.shape[1] * margin),
            int(work.shape[0] * margin),
            int(work.shape[1] * (1 - 2 * margin)),
            int(work.shape[0] * (1 - 2 * margin)))
    
    cv2.grabCut(work, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Get foreground mask
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Find largest foreground contour
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (OUT_W, OUT_H))
    
    largest = max(contours, key=cv2.contourArea)
    
    # Get 4 corners
    peri = cv2.arcLength(largest, True)
    corners = cv2.approxPolyDP(largest, 0.02 * peri, True)
    
    if len(corners) == 4:
        note = _perspective_transform(image, corners, scale, OUT_W, OUT_H)
    else:
        x, y, cw, ch = cv2.boundingRect(largest)
        if scale < 1.0:
            x, y, cw, ch = int(x/scale), int(y/scale), int(cw/scale), int(ch/scale)
        note = image[y:y+ch, x:x+cw]
        note = cv2.resize(note, (OUT_W, OUT_H))
    
    # Convert to grayscale
    if len(note.shape) == 3:
        note = cv2.cvtColor(note, cv2.COLOR_BGR2GRAY)
    
    return note


def _perspective_transform(image, corners, scale, out_w, out_h):
    pts = corners.reshape(-1, 2).astype(np.float32)
    
    if scale < 1.0:
        pts = pts / scale
    
    # Order points
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    
    # Shrink inward
    centroid = ordered.mean(axis=0)
    shrink = 0.04
    ordered = ordered + shrink * (centroid - ordered)
    
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    
    return warped


def segment_note_simple(image, use_clahe=False):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply preprocessing
    if use_clahe:
        gray = preprocessing_CLAHE(gray)
    else:
        gray = preprocessing_global(gray)
    
    # Edge detection
    edges = detect_edges_canny(gray)
    
    # Find largest contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(gray.shape[1] - x, w + 2 * padding)
    h = min(gray.shape[0] - y, h + 2 * padding)
    
    return gray[y:y+h, x:x+w]


# Comparison Function
def segment_note_with_comparison(image, use_clahe=False):
    
    if image is None or image.size == 0:
        return None
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply preprocessing
    if use_clahe:
        gray = preprocessing_CLAHE(gray)
    else:
        gray = preprocessing_global(gray)
    
    results = {}
    
    # Run Canny segmentation
    canny_edges = detect_edges_canny(gray)
    canny_contour = _find_largest_contour(canny_edges)
    canny_cropped = _crop_from_contour(gray, canny_contour) if canny_contour is not None else None
    
    results['canny'] = {
        'edges': canny_edges,
        'contour': canny_contour,
        'cropped': canny_cropped
    }
    
    # Run Sobel segmentation
    sobel_edges = detect_edges_sobel(gray)
    sobel_contour = _find_largest_contour(sobel_edges)
    sobel_cropped = _crop_from_contour(gray, sobel_contour) if sobel_contour is not None else None
    
    results['sobel'] = {
        'edges': sobel_edges,
        'contour': sobel_contour,
        'cropped': sobel_cropped
    }
    
    return results


def _find_largest_contour(edge_map):
   
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _crop_from_contour(image, contour, padding=10):
   
    x, y, w, h = cv2.boundingRect(contour)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    return image[y:y+h, x:x+w]