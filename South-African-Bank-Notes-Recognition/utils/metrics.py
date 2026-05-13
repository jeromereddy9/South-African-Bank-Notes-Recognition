import os
import re
import cv2
import random
import numpy as np

banknote_dataset_path = "Dataset\Banknote_Dataset_(2005-2023)"
pattern = r'R\d{2,3}_(Front|Back)_\d{1,3}_\d{4}.png'


def validate_dataset_filenames(file_path,pattern):
    total_files = 0
    valid_files = 0
    invalid_names = []

    for filename in os.listdir(file_path):
        total_files += 1
        if re.search(pattern,filename):
            valid_files += 1
        else:
            invalid_names.append(filename)

    if total_files == valid_files:
        print("All files names valid.")
        return True
    else:
        print(f"Invalid file names detected: \n{invalid_names}")
        return False



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







image = cv2.imread('Dataset\Banknote_Dataset_(2005-2023)\R10_Back_0_2005.png')

a_image = generate_augmented_view(image)
cv2.namedWindow("augmented", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("augmented", 800, 600)
cv2.imshow("augmented",a_image)
cv2.waitKey(0) 
cv2.destroyAllWindows()

