import cv2
import random
import numpy as np

def generate_augmented_view(img, apply_brightness=True, apply_contrast=True, 
                            apply_flip=False, apply_blur=False, apply_noise=False):
    """
    Generate an augmented view of a banknote image.
    
    Args:
        img: Grayscale image
        apply_brightness: Random brightness adjustment
        apply_contrast: Random contrast adjustment
        apply_flip: Random horizontal flip
        apply_blur: Random Gaussian blur
        apply_noise: Random Gaussian noise
    
    Returns:
        Augmented image
    """
    if img is None:
        return None
    
    result = img.copy()
    
    # 1. Random horizontal flip (50% chance)
    if apply_flip and random.random() > 0.5:
        result = cv2.flip(result, 1)
    
    # 2. Rotation and scaling (your existing code, improved)
    random_rotation_value = random.randint(0, 359)
    random_scaling_value = round(random.uniform(0.7, 1.3), 2)  # Wider scale range

    (h, w) = result.shape[:2]
    center = (w // 2, h // 2)
    
    matrix = cv2.getRotationMatrix2D(center, random_rotation_value, random_scaling_value)
    
    cos = np.abs(matrix[0, 0].item())
    sin = np.abs(matrix[0, 1].item())
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    result = cv2.warpAffine(result, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)
    
    # 3. Brightness adjustment
    if apply_brightness:
        brightness = random.uniform(0.6, 1.4)  # ±40%
        result = cv2.convertScaleAbs(result, alpha=1.0, beta=(brightness - 1.0) * 100)
    
    # 4. Contrast adjustment
    if apply_contrast:
        contrast = random.uniform(0.7, 1.3)  # ±30%
        result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)
    
    # 5. Gaussian blur (30% chance)
    if apply_blur and random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    
    # 6. Gaussian noise (30% chance)
    if apply_noise and random.random() < 0.3:
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, result.shape).astype(np.uint8)
        result = cv2.add(result, noise)
    
    # 7. Final resize to 224x224
    result = cv2.resize(result, (224, 224))
    
    return result


def generate_enhanced_references(img, denomination=None, num_variations=10):
    """
    Generate multiple reference variations of an image for SIFT database.
    Uses more variations for problematic denominations (R50).
    
    Args:
        img: Grayscale image
        denomination: Denomination label (e.g., "R50")
        num_variations: Base number of variations
    
    Returns:
        List of reference images
    """
    references = []
    
    # Always include original resized
    references.append(cv2.resize(img, (224, 224)))
    
    # Determine augmentation intensity based on denomination
    if denomination == "R50":
        # R50 needs more help (confused with R100)
        intensity = "high"
        actual_variations = num_variations * 2
    else:
        intensity = "normal"
        actual_variations = num_variations
    
    # Generate variations
    for _ in range(actual_variations):
        if intensity == "high":
            # More aggressive augmentation for R50
            aug = generate_augmented_view(
                img, 
                apply_brightness=True,
                apply_contrast=True,
                apply_flip=True,
                apply_blur=True,
                apply_noise=True
            )
        else:
            # Standard augmentation for others
            aug = generate_augmented_view(
                img,
                apply_brightness=True,
                apply_contrast=True,
                apply_flip=False,
                apply_blur=False,
                apply_noise=False
            )
        references.append(aug)
    
    return references