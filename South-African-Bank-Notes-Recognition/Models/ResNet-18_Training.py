import os, sys, copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ResNet18", os.path.join(PROJECT_ROOT, "Models", "ResNet-18.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_resnet18    = _mod.build_resnet18
unfreeze_backbone = _mod.unfreeze_backbone

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CONFIG = {
    "dataset_root"   : os.path.join(PROJECT_ROOT, "Dataset", "raw", "Banknote_Dataset_(2005-2023)"),
    "num_workers"    : 0,
    "pretrained"     : True,
    "freeze_epochs"  : 5,
    "total_epochs"   : 100,          
    "lr_head"        : 1e-3,
    "lr_finetune"    : 5e-5,        
    "batch_size"     : 16,
    "weight_decay"   : 1e-4,
    "label_smoothing": 0.1,        
    "checkpoint_dir" : os.path.join(PROJECT_ROOT, "checkpoints"),
}

def normalise_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std

def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for image_tensors, label_tensors in loader:
        image_tensors = torch.stack([normalise_tensor(t) for t in image_tensors]).to(device)
        label_tensors = label_tensors.to(device)
        optimiser.zero_grad()
        outputs = model(image_tensors)
        loss    = criterion(outputs, label_tensors)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * image_tensors.size(0)
        _, predicted  = torch.max(outputs, 1)
        correct      += (predicted == label_tensors).sum().item()
        total        += image_tensors.size(0)
    return running_loss / total if total else 0.0, correct / total if total else 0.0

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for image_tensors, label_tensors in loader:
            image_tensors = torch.stack([normalise_tensor(t) for t in image_tensors]).to(device)
            label_tensors = label_tensors.to(device)
            outputs = model(image_tensors)
            loss    = criterion(outputs, label_tensors)
            running_loss += loss.item() * image_tensors.size(0)
            _, predicted  = torch.max(outputs, 1)
            correct      += (predicted == label_tensors).sum().item()
            total        += image_tensors.size(0)
    return running_loss / total if total else 0.0, correct / total if total else 0.0

def train_fold(fold_idx, train_idx, val_idx, config, device):
    from Data.dataloader import BanknoteDataset
    
    print("\n" + "=" * 60)
    print(f"FOLD {fold_idx + 1}/5")
    print("=" * 60)

    # Bypass Validation Augmentation Trap
    train_dataset_full = BanknoteDataset(root=config["dataset_root"], augment=True, colour=True)
    val_dataset_full = BanknoteDataset(root=config["dataset_root"], augment=False, colour=True)

    train_set = Subset(train_dataset_full, train_idx)
    val_set = Subset(val_dataset_full, val_idx)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader   = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    model = build_resnet18(pretrained=config["pretrained"], freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Phase 1
    optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr_head"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config["freeze_epochs"], eta_min=1e-5)
    
    for _ in range(config["freeze_epochs"]):
        train_one_epoch(model, train_loader, criterion, optimiser, device)
        _, va = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        if va > best_val_acc:
            best_val_acc = va
            best_model_wts = copy.deepcopy(model.state_dict())

    # Phase 2
    model = unfreeze_backbone(model)
    remaining = config["total_epochs"] - config["freeze_epochs"]
    optimiser = optim.Adam(model.parameters(), lr=config["lr_finetune"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=remaining, eta_min=1e-6)
    
    for _ in range(remaining):
        train_one_epoch(model, train_loader, criterion, optimiser, device)
        _, va = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        if va > best_val_acc:
            best_val_acc = va
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Fold {fold_idx + 1} Best Validation Accuracy: {best_val_acc*100:.1f}%")
    return best_val_acc, best_model_wts


if __name__ == "__main__":
    from Data.dataloader import BanknoteDataset
    print("ResNet-18 5-Fold CV Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_dataset = BanknoteDataset(root=CONFIG["dataset_root"], augment=False)
    indices = np.arange(len(dummy_dataset))
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    global_best_acc = 0.0
    global_best_checkpoint = {}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        best_acc, best_wts = train_fold(fold, train_idx, val_idx, CONFIG, device)
        fold_results.append(best_acc)
        
        # Track the absolute best model across all folds
        if best_acc > global_best_acc:
            global_best_acc = best_acc
            global_best_checkpoint = {
                'model_state_dict': best_wts,
                'val_acc': best_acc,
                'fold': fold + 1
            }
            
    # Save the global winner to the disk
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    save_path = os.path.join(CONFIG["checkpoint_dir"], "resnet18_best.pth")
    torch.save(global_best_checkpoint, save_path)
        
    print("\n" + "=" * 60)
    print("5-FOLD CV COMPLETE")
    print(f"Average Accuracy: {np.mean(fold_results)*100:.1f}% ± {np.std(fold_results)*100:.1f}%")
    print(f"Highest performing model (Fold {global_best_checkpoint['fold']}) saved to: {save_path}")