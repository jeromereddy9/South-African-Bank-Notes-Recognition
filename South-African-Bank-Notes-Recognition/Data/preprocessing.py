import cv2
import numpy as np

def equalize_histogram(image):
    """
    Apply global histogram equalization to each channel of RGB.
    """
    if image is None:
        return None
    
    # If color (3 channels), process each channel separately
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to YUV (equalize only luminance, preserve color)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # Grayscale
        return cv2.equalizeHist(image)


def equalize_clahe(image, clip_limit=2.0, grid_size=(8,8)):
    """
    Apply CLAHE to RGB image (on luminance channel).
    """
    if image is None:
        return None
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to LAB (apply CLAHE to L channel)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale
        return clahe.apply(image)


def load_image(image_path):
    """Load image in color (BGR)."""
    return cv2.imread(image_path)


def apply_gaussian_smoothing(image, kernel_size=5):
    """Apply Gaussian blur (works on RGB and grayscale)."""
    if image is None:
        return None
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def preprocessing_global(image):
    """Global equalization preprocessing (preserves color)."""
    if image is None:
        return None
    smoothed = apply_gaussian_smoothing(image, kernel_size=3)
    enhanced = equalize_histogram(smoothed)
    return enhanced


def preprocessing_CLAHE(image):
    """CLAHE preprocessing (preserves color)."""
    if image is None:
        return None
    smoothed = apply_gaussian_smoothing(image, kernel_size=3)
    enhanced = equalize_clahe(smoothed, clip_limit=2.0, grid_size=(8,8))
    return enhanced

def preprocess_for_model(image_path, segment=True):
    """
    Standardized engine for real-time evaluation or training extraction.
    Loads image -> Isolates banknote -> Returns clean BGR color image.
    """
    import cv2
    from Data.segmentation import segment_note 
    
    # 1. Read Image in Color (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image path: {image_path}")
        
    # 2. Dynamic Structural Isolation
    if segment:
        img_processed = segment_note(img_bgr)
        # Safety fallback: If GrabCut totally fails, use the original image
        if img_processed is None:
            img_processed = img_bgr
    else:
        img_processed = img_bgr
        
    return img_processed

def enhance_resolution(image, target_width=640):
    """
    Checks if the segmented banknote is too small. 
    If it is, it upscales it and applies a sharpening filter to restore edge data.
    """
    if image is None:
        return None
        
    h, w = image.shape[:2]
    
    # If the image is smaller than our target width (e.g., it was far away)
    if w < target_width:
        # 1. Upscale using CUBIC interpolation (best for adding artificial pixels)
        scale = target_width / w
        new_h = int(h * scale)
        upscaled = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Apply a Sharpening Kernel (Unsharp Masking)
        # This pushes edge contrast up, which SIFT and CNNs love.
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
        
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        return sharpened
        
    # If the image is already large enough, just return it as-is
    return image