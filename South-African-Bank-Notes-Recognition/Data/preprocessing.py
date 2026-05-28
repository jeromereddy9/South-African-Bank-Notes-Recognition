import cv2
import numpy as np

def equalize_histogram(image):

    if image is None:
        return None
    
    # If color (3 channels), process luminance only to preserve color
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to YUV (Y = luminance, UV = color)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Equalize only the luminance channel
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        # Convert back to BGR
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # Grayscale image - equalize directly
        return cv2.equalizeHist(image)


def equalize_clahe(image, clip_limit=1.5, grid_size=(10, 10)):

    if image is None:
        return None
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to LAB color space (L = lightness)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Apply CLAHE only to L channel (preserves color)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale - apply directly
        return clahe.apply(image)


def enhance_resolution(image, target_width=640):

    if image is None:
        return None
        
    h, w = image.shape[:2]
    
    # Calculate new height to maintain aspect ratio
    scale = target_width / w
    new_h = int(h * scale)
    
    # Upscale using cubic interpolation (smoother than linear)
    upscaled = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_CUBIC)
        
    # Mild sharpening kernel (softer to avoid creating halos)
    kernel = np.array([[ 0, -0.5,  0],
                       [-0.5,  3.0, -0.5],
                       [ 0, -0.5,  0]])
        
    sharpened = cv2.filter2D(upscaled, -1, kernel)
        
    return sharpened


def load_image(image_path):
    return cv2.imread(image_path)


def apply_gaussian_smoothing(image, kernel_size=5):

    if image is None:
        return None
    # Ensure kernel size is odd (required by GaussianBlur)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def preprocessing_global(image):

    if image is None:
        return None
    smoothed = apply_gaussian_smoothing(image, kernel_size=3)
    enhanced = equalize_histogram(smoothed)
    return enhanced


def preprocessing_CLAHE(image):

    if image is None:
        return None
    smoothed = apply_gaussian_smoothing(image, kernel_size=3)
    enhanced = equalize_clahe(smoothed, clip_limit=2.0, grid_size=(8,8))
    return enhanced


def preprocess_for_model(image_path, segment=True):

    import cv2
    from Data.segmentation import segment_note 
    
    # 1. Read image in color 
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image path: {image_path}")
        
    # 2. Apply segmentation to isolate the banknote 
    if segment:
        img_processed = segment_note(img_bgr)
        # Fallback
        if img_processed is None:
            img_processed = img_bgr
    else:
        img_processed = img_bgr
        
    return img_processed