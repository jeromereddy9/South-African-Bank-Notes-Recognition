import cv2
import numpy as np
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import your existing segmentation function
from Data.segmentation import segment_note
from Data.preprocessing import enhance_resolution


def process_and_snapshot(image_path, output_dir="Results/Pipeline"):
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"Generating snapshots for: {base_name}")
    
    # Original Image
    # Load the original image 
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not read image at {image_path}")
        return
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_1_Original.jpg"), original)

    # Edge Detection 
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # Smooth to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 40, 120)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_2_CannyEdges.jpg"), edges)

    # Segmentation
    # Extract the banknote from background and straighten it
    segmented = segment_note(original)
    if segmented is not None:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_3_Segmented.jpg"), segmented)
    else:
        print("Segmentation failed. Stopping snapshot pipeline.")
        return

    # Enhancement 
    # Improve image quality for better feature detection
    enhanced = enhance_resolution(segmented)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_4_Enhanced.jpg"), enhanced)

    # 5. SIFT Feature Extraction 
    # Create SIFT detector
    sift = cv2.SIFT_create()
    # Convert enhanced image to grayscale (SIFT requires single channel)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(enhanced_gray, None)
    
    # Draw keypoints on the image for visualization
    sift_img = cv2.drawKeypoints(enhanced_gray, keypoints, None, 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_5_SIFT_Features.jpg"), sift_img)

    print(f"Success! Snapshots saved to /{output_dir}\n")


if __name__ == "__main__":
    # Path to test image 
    test_image_path = "Dataset/Test/R50_1.png" 
    
    if os.path.exists(test_image_path):
        process_and_snapshot(test_image_path)
    else:
        print(f"Please ensure '{test_image_path}' is in the correct folder.")