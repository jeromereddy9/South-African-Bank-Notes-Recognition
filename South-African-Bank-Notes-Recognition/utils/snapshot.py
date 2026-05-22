"""
Snapshot utilities for saving report images at each pipeline stage.
"""

import cv2
import os
from pathlib import Path


def save_snapshot(image, filename, folder="snapshots"):
    """
    Save an image to the snapshots folder for the report.
    
    Args:
        image: Image to save (numpy array)
        filename: Name of the file (e.g., "1_original.png")
        folder: Output folder (default: "snapshots")
    
    Returns:
        Full path to saved file
    """
    # Create folder if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    # Full path
    filepath = os.path.join(folder, filename)
    
    # Save image
    cv2.imwrite(filepath, image)
    
    return filepath


def save_comparison_grid(images, titles, filename, folder="South-African-Bank-Notes-Recognition\Results\Snapshot", cols=2):
    """
    Save a grid of images side-by-side for comparison.
    Useful for showing global vs CLAHE, or Canny vs Sobel.
    
    Args:
        images: List of images (numpy arrays)
        titles: List of titles for each image
        filename: Output filename
        folder: Output folder (default: "snapshots")
        cols: Number of columns in grid (default: 2)
    """
    import matplotlib.pyplot as plt
    
    rows = (len(images) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(len(images), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save
    Path(folder).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath