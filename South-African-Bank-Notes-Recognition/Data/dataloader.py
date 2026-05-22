import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from data.preprocessing import preprocessing_global, preprocessing_CLAHE
from data.segmentation import segment_note, segment_note_simple
from data.augmentation import get_augmented_view


class BanknoteDataset(Dataset):
    
    def __init__(self, image_paths, labels, is_training=True, 
                 use_clahe=False, use_robust_segmentation=True):
        
        self.image_paths = image_paths
        self.labels = labels
        self.is_training = is_training
        self.use_clahe = use_clahe
        self.use_robust_segmentation = use_robust_segmentation
        
        # Output size 
        self.output_size = (224, 224)  # ResNet expects 224x224
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
      
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Preprocessing
        if self.use_clahe:
            gray = preprocessing_CLAHE(gray)
        else:
            gray = preprocessing_global(gray)
        
        # Segmentation 
        if self.use_robust_segmentation:
            note = segment_note(gray, use_clahe=self.use_clahe)
           
            if note is None or note.size == 0:
                note = segment_note_simple(gray, use_clahe=self.use_clahe)
        else:
            note = segment_note_simple(gray, use_clahe=self.use_clahe)
        
        if note is None or note.size == 0:
  
            note = gray
        
        # Resize to fixed size
        note = cv2.resize(note, self.output_size)
        
        # Augmentation (training only)
        if self.is_training:
            note = get_augmented_view(note)
        
        # Convert to tensor and normalize to [0, 1]
        note_tensor = torch.from_numpy(note).float().unsqueeze(0) / 255.0  # (1, H, W)
        
        # Label tensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return note_tensor, label_tensor


def create_dataloaders(image_paths, labels, batch_size=32, train_split=0.8,
                       use_clahe=False, use_robust_segmentation=True, 
                       random_seed=42):
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create list of indices
    indices = np.arange(len(image_paths))
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Split
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Create datasets
    train_dataset = BanknoteDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        is_training=True,
        use_clahe=use_clahe,
        use_robust_segmentation=use_robust_segmentation
    )
    
    test_dataset = BanknoteDataset(
        [image_paths[i] for i in test_indices],
        [labels[i] for i in test_indices],
        is_training=False,  
        use_clahe=use_clahe,
        use_robust_segmentation=use_robust_segmentation
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_indices, test_indices


def create_kfold_dataloaders(image_paths, labels, n_folds=5, batch_size=32,
                             use_clahe=False, use_robust_segmentation=True,
                             random_seed=42):
    
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    folds = []
    
    for train_indices, test_indices in kf.split(image_paths):
        # Create datasets for this fold
        train_dataset = BanknoteDataset(
            [image_paths[i] for i in train_indices],
            [labels[i] for i in train_indices],
            is_training=True,
            use_clahe=use_clahe,
            use_robust_segmentation=use_robust_segmentation
        )
        
        test_dataset = BanknoteDataset(
            [image_paths[i] for i in test_indices],
            [labels[i] for i in test_indices],
            is_training=False,
            use_clahe=use_clahe,
            use_robust_segmentation=use_robust_segmentation
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        folds.append((train_loader, test_loader, train_indices, test_indices))
    
    return folds



# Helper function to load dataset

def load_dataset_from_folder(folder_path, label_mapping):
  
    image_paths = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Extract denomination from filename 
            parts = filename.split('_')
            if len(parts) >= 1:
                denom_str = parts[0]  
                
                # Convert denomination to label
                for pattern, label in label_mapping.items():
                    if denom_str == pattern:
                        image_paths.append(os.path.join(folder_path, filename))
                        labels.append(label)
                        break
    
    return image_paths, labels
