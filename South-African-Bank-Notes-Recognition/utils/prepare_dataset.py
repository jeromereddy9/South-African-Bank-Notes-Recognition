import os
import sys
import cv2
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import segmentation function
from Data.segmentation import segment_note

# Define input and output paths
RAW_PATH = "Dataset/raw/Banknote_Dataset_(2005-2023)"   # Original images
SAVE_PATH = "Dataset/segmented"                         # Segmented images

# Create output directory if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def run_presegmentation():
    
    # Get all PNG files from raw dataset
    files = [f for f in os.listdir(RAW_PATH) if f.lower().endswith('.png')]
    print(f"Segmenting {len(files)} images...")
    
    # Process each image with progress bar
    for fn in tqdm(files):
        img_path = os.path.join(RAW_PATH, fn)
        img = cv2.imread(img_path)
        
        if img is None:
            continue  # Skip if image cannot be read
            
        # Apply segmentation to extract the banknote
        segmented = segment_note(img)
        
        if segmented is not None:
            # Save segmented image if successful
            cv2.imwrite(os.path.join(SAVE_PATH, fn), segmented)
        else:
            # Fallback: save original if segmentation fails
            cv2.imwrite(os.path.join(SAVE_PATH, fn), img)


if __name__ == "__main__":
    run_presegmentation()