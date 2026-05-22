import cv2

def equalize_histogram(image):
    #Apply global histogram equalization to improve contrast across entire image
    if image is None:
        return None
    return cv2.equalizeHist(image)


def equalize_clahe(image, clip_limit=2.0, grid_size=(8,8)):
    #Apply CLAHE  equalization
    if image is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def load_image_in_grayscale(image_path):
    #Load image from disk and convert to grayscale
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return grayscale_image


def apply_gaussian_smoothing(image, kernel_size=5):
    #Reduce noise and smooth image using Gaussian blur.Kernel size must be odd (auto-corrected if even).
    if image is None:
        return None
    # Ensure kernel size is odd (required by GaussianBlur)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return smoothed_image

def preprocessing_global(image):
    if image is None:
        return None
    #Smooths first to prevent amplifying high-frequency background noise
    smoothed = apply_gaussian_smoothing(image, kernel_size=3)
    #Maximize global contrast
    enhanced = equalize_histogram(smoothed)
    return enhanced

def preprocessing_CLAHE(image):
    if image is None:
        return None
    #Smooths to suppress tiny speckle noise
    smoothed = apply_gaussian_smoothing(image, kernel_size=3)
    #Locally enhance text, features, and invariant properties
    enhanced = equalize_clahe(smoothed, clip_limit=2.0, grid_size=(8,8))
    return enhanced

