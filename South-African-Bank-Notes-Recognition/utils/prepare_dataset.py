import os,sys
import cv2
from tqdm import tqdm
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Data.segmentation import segment_note # Using your robust script

RAW_PATH = "Dataset/raw/Banknote_Dataset_(2005-2023)"
SAVE_PATH = "Dataset/segmented"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def run_presegmentation():
    files = [f for f in os.listdir(RAW_PATH) if f.lower().endswith('.png')]
    print(f"Segmenting {len(files)} images...")
    
    for fn in tqdm(files):
        img_path = os.path.join(RAW_PATH, fn)
        img = cv2.imread(img_path)
        
        if img is None: continue
            
        # Use your robust segmentation function
        segmented = segment_note(img)
        
        if segmented is not None:
            # Save the clean output
            cv2.imwrite(os.path.join(SAVE_PATH, fn), segmented)
        else:
            # Fallback: copy original if segmentation fails
            cv2.imwrite(os.path.join(SAVE_PATH, fn), img)

if __name__ == "__main__":
    run_presegmentation()