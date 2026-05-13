import cv2
import random
import numpy as np

def generate_augmented_view(img):
    random_rotation_value = random.randint(0, 359)
    random_scaling_value = round(random.uniform(0.8, 1.2), 2)

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    
    matrix = cv2.getRotationMatrix2D(center, random_rotation_value, random_scaling_value)
    
    cos = np.abs(matrix[0, 0].item())
    sin = np.abs(matrix[0, 1].item())
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    augmented_image = cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return augmented_image


