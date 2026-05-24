import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm
from Models.SimCLR import get_simclr_model, get_linear_probe, ContrastiveLoss
from Data.dataloader import BanknoteDataset

CONFIG = {
    "dataset_root": os.path.join("Dataset", "raw", "Banknote_Dataset_(2005-2023)"),
    "num_workers": 0,
    
    # SimCLR training
    "projection_dim": 128,
    "temperature": 0.5,  
    "contrastive_epochs": 100,
    "contrastive_lr": 3e-4,  
    
    # Linear probe training
    "probe_epochs": 50,
    "probe_lr": 1e-2,
    
    "batch_size": 32,
    "weight_decay": 1e-4,
    
    "checkpoint_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"),
    "best_model_name": "simclr_best.pth",
    "curve_save_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_curves_simclr.png"),
}

CLASS_LABELS = ["R10", "R20", "R50", "R100", "R200"]
NUM_CLASSES = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ContrastiveAugmentation:
    def __init__(self, size=224, s=1.0):
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s
        )
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    def __call__(self, x):
        return self.transform(x), self.transform(x)


class ContrastiveDatasetWrapper(Dataset):
    def __init__(self, base_dataset, augmentation):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        view1, view2 = self.augmentation(image)
        return view1, view2, label


def train_contrastive_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, num_batches = 0.0, 0
    for view1, view2, _ in tqdm(loader, desc="Contrastive Training", leave=False):
        view1, view2 = view1.to(device), view2.to(device)
        batch_size = view1.shape[0]
        
        emb1, emb2 = model(view1), model(view2)
        loss = criterion(emb1, emb2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        num_batches += batch_size
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_probe_epoch(model, probe, loader, criterion, optimizer, device):
    model.eval()
    probe.train()
    total_loss, correct, total = 0.0, 0, 0
    # Uses standard dataloader (images, labels)
    for images, labels in tqdm(loader, desc="Probe Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = model.get_features(images)
        outputs = probe(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0


def evaluate_probe(model, probe, loader, device):
    model.eval()
    probe.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = model.get_features(images)
            outputs = probe(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0


def train_fold(fold_idx, train_idx, val_idx, config, device):
    print("\n" + "=" * 60)
    print(f"FOLD {fold_idx + 1}/5")
    print("=" * 60)
    
    # Fix Validation Trap: Clean subsets for train and validation
    train_dataset_base = BanknoteDataset(root=config["dataset_root"], augment=False, colour=True)
    val_dataset_clean = BanknoteDataset(root=config["dataset_root"], augment=False, colour=True)
    
    train_subset = Subset(train_dataset_base, train_idx)
    val_subset = Subset(val_dataset_clean, val_idx)
    
    # Phase 1 Dataloader (Contrastive)
    contrastive_aug = ContrastiveAugmentation(size=224, s=1.0)
    contrastive_set = ContrastiveDatasetWrapper(train_subset, contrastive_aug)
    train_loader_simclr = DataLoader(contrastive_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    
    # Phase 2 Dataloaders (Linear Probe uses standard images)
    train_dataset_probe = BanknoteDataset(root=config["dataset_root"], augment=True, colour=True)
    train_subset_probe = Subset(train_dataset_probe, train_idx)
    train_loader_probe = DataLoader(train_subset_probe, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader_probe = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    # Initialize Models
    model = get_simclr_model(projection_dim=config["projection_dim"], pretrained=True, input_channels=3).to(device)
    probe = get_linear_probe(input_dim=512, num_classes=NUM_CLASSES).to(device)
    
    # --- PHASE 1: Contrastive ---
    criterion_simclr = ContrastiveLoss(temperature=config['temperature'])
    opt_simclr = optim.Adam(model.parameters(), lr=config['contrastive_lr'], weight_decay=config['weight_decay'])
    scheduler_simclr = optim.lr_scheduler.CosineAnnealingLR(opt_simclr, T_max=config['contrastive_epochs'], eta_min=1e-6)
    
    for epoch in range(1, config['contrastive_epochs'] + 1):
        loss = train_contrastive_epoch(model, train_loader_simclr, criterion_simclr, opt_simclr, device)
        scheduler_simclr.step()
    
    # --- PHASE 2: Linear Probe ---
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    criterion_probe = nn.CrossEntropyLoss()
    opt_probe = optim.Adam(probe.parameters(), lr=config['probe_lr'], weight_decay=config['weight_decay'])
    scheduler_probe = optim.lr_scheduler.CosineAnnealingLR(opt_probe, T_max=config['probe_epochs'], eta_min=1e-6)
    
    best_val_acc = 0.0
    best_backbone_wts = None
    best_probe_wts = None
    
    for epoch in range(1, config['probe_epochs'] + 1):
        train_probe_epoch(model, probe, train_loader_probe, criterion_probe, opt_probe, device)
        _, val_acc = evaluate_probe(model, probe, val_loader_probe, device)
        scheduler_probe.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_backbone_wts = copy.deepcopy(model.backbone.state_dict())
            best_probe_wts = copy.deepcopy(probe.state_dict())
            
    print(f"Fold {fold_idx + 1} Best Validation Accuracy: {best_val_acc*100:.1f}%")
    
    # Return the accuracy and the specific state dictionaries from this fold's peak
    return best_val_acc, best_backbone_wts, best_probe_wts


if __name__ == "__main__":
    print("SimCLR 5-Fold CV Training")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_dataset = BanknoteDataset(root=CONFIG["dataset_root"], augment=False)
    indices = np.arange(len(dummy_dataset))
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    global_best_acc = 0.0
    global_best_checkpoint = {}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        best_acc, backbone_wts, probe_wts = train_fold(fold, train_idx, val_idx, CONFIG, device)
        fold_results.append(best_acc)
        
        # Track the absolute best model across all folds
        if best_acc > global_best_acc:
            global_best_acc = best_acc
            global_best_checkpoint = {
                'backbone_state_dict': backbone_wts,
                'probe_state_dict': probe_wts,
                'val_acc': best_acc,
                'fold': fold + 1
            }
            
    # Save the global winner to the disk
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    save_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["best_model_name"])
    torch.save(global_best_checkpoint, save_path)
        
    print("\n" + "=" * 60)
    print("5-FOLD CV COMPLETE")
    print(f"Average Accuracy: {np.mean(fold_results)*100:.1f}% ± {np.std(fold_results)*100:.1f}%")
    print(f"Highest performing model (Fold {global_best_checkpoint['fold']}) saved to: {save_path}")