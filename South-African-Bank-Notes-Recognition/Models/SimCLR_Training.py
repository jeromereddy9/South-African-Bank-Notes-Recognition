import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from Models.SimCLR import get_simclr_model, get_linear_probe, ContrastiveLoss
from Data.dataloader import BanknoteDataset


CONFIG = {
    # Data
    "dataset_root": os.path.join("Dataset", "raw", "Banknote_Dataset_(2005-2023)"),
    "val_split": 0.3,
    "num_workers": 0,
    
    # SimCLR training (Phase 1 - contrastive)
    "projection_dim": 128,
    "temperature": 0.2,
    "contrastive_epochs": 100,
    "contrastive_lr": 1e-3,
    
    # Linear probe training (Phase 2 - supervised)
    "probe_epochs": 50,
    "probe_lr": 1e-2,
    
    # General
    "batch_size": 32,
    "weight_decay": 1e-4,
    
    # Model saving
    "checkpoint_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"),
    "best_model_name": "simclr_best.pth",
    "curve_save_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_curves_simclr.png"),
    
    # Preprocessing options
    "use_clahe": False,
    "use_robust_segmentation": True,
}

# Label mapping (for reference)
CLASS_LABELS = ["R10", "R20", "R50", "R100", "R200"]
NUM_CLASSES = 5


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_training_curves(history, save_path):
    """Save loss and accuracy curves for the report."""
    epochs = range(1, len(history["contrastive_loss"]) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Phase 1: Contrastive Loss
    ax1.plot(epochs, history["contrastive_loss"], marker="o", color="blue", linewidth=2)
    ax1.set_title("Phase 1: Contrastive Loss (SimCLR)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    
    # Phase 2: Probe Loss
    probe_epochs = range(1, len(history["probe_train_loss"]) + 1)
    ax2.plot(probe_epochs, history["probe_train_loss"], label="Train Loss", marker="o")
    ax2.plot(probe_epochs, history["probe_val_loss"], label="Val Loss", marker="s")
    ax2.set_title("Phase 2: Linear Probe Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)
    
    # Phase 2: Probe Accuracy
    ax3.plot(probe_epochs, [a * 100 for a in history["probe_train_acc"]], label="Train Acc", marker="o")
    ax3.plot(probe_epochs, [a * 100 for a in history["probe_val_acc"]], label="Val Acc", marker="s")
    ax3.set_title("Phase 2: Linear Probe Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.legend()
    ax3.grid(True)
    
    # Summary: Best accuracy
    best_acc = max(history["probe_val_acc"]) * 100
    ax4.text(0.5, 0.5, f"Best Validation Accuracy: {best_acc:.1f}%", 
             fontsize=16, ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title("Training Summary")
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved -> {save_path}")


def train_contrastive_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, _ in tqdm(loader, desc="Contrastive Training", leave=False):
        images = images.to(device)
        batch_size = images.shape[0]
        
        emb1 = model(images)
        emb2 = model(images)  
        
        loss = criterion(emb1, emb2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        num_batches += batch_size
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_contrastive(model, train_loader, config, device):
   
 
    print("PHASE 1: CONTRASTIVE LEARNING (SimCLR)")
    print(f"Epochs: {config['contrastive_epochs']} | Learning rate: {config['contrastive_lr']}")
    print(f"Projection dim: {config['projection_dim']} | Temperature: {config['temperature']}")
    
    model = model.to(device)
    criterion = ContrastiveLoss(temperature=config['temperature'])
    optimizer = optim.Adam(model.parameters(), 
                           lr=config['contrastive_lr'], 
                           weight_decay=config['weight_decay'])
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['contrastive_epochs'], eta_min=1e-5
    )
    
    loss_history = []
    
    for epoch in range(1, config['contrastive_epochs'] + 1):
        loss = train_contrastive_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        loss_history.append(loss)
        
        print(f"Epoch {epoch:02d}/{config['contrastive_epochs']:02d} | Contrastive Loss: {loss:.4f}")
    
    return model, loss_history




def train_probe_epoch(model, probe, loader, criterion, optimizer, device):

    model.eval()  
    probe.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Probe Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        # Get frozen features from backbone
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
    
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            features = model.get_features(images)
            outputs = probe(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return (total_loss / total if total > 0 else 0.0,
            correct / total if total > 0 else 0.0)


def train_linear_probe(model, train_loader, val_loader, config, device, checkpoint_dir, best_model_path):
   
   
    print("PHASE 2: LINEAR PROBE TRAINING")
    print(f"Epochs: {config['probe_epochs']} | Learning rate: {config['probe_lr']}")
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Create linear probe
    probe = get_linear_probe(input_dim=512, num_classes=NUM_CLASSES)
    probe = probe.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), 
                           lr=config['probe_lr'], 
                           weight_decay=config['weight_decay'])
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['probe_epochs'], eta_min=1e-6
    )
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_probe_weights = copy.deepcopy(probe.state_dict())
    best_backbone_weights = copy.deepcopy(model.backbone.state_dict())
    
    for epoch in range(1, config['probe_epochs'] + 1):
        # Training
        train_loss, train_acc = train_probe_epoch(model, probe, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = evaluate_probe(model, probe, val_loader, device)
        
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:02d}/{config['probe_epochs']:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")
        
        # Save if better
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_probe_weights = copy.deepcopy(probe.state_dict())
            best_backbone_weights = copy.deepcopy(model.backbone.state_dict())
            
            torch.save({
                "backbone_state_dict": best_backbone_weights,
                "probe_state_dict": best_probe_weights,
                "val_acc": best_val_acc,
                "val_loss": best_val_loss,
                "epoch": epoch,
                "config": config
            }, best_model_path)
            print(f"Best model saved (val_acc={best_val_acc*100:.1f}%)")
    
    # Load best weights
    probe.load_state_dict(best_probe_weights)
    model.backbone.load_state_dict(best_backbone_weights)
    
    return model, probe, history, best_val_acc


def train(dataset, config: dict = None):
   
    cfg = {**CONFIG, **(config or {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("SIMCLR TRAINING FOR BANKNOTE RECOGNITION")
    print(f"Training on  : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Split dataset into train/val
    n_total = len(dataset)
    n_val = max(1, int(n_total * cfg["val_split"]))
    n_train = n_total - n_val
    
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"])
    
    print(f"\nTrain samples: {n_train}  |  Val samples: {n_val}\n")
    
    # Create checkpoint directory
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_model_path = os.path.join(cfg["checkpoint_dir"], cfg["best_model_name"])
    
    # Create model
    model = get_simclr_model(projection_dim=cfg["projection_dim"], pretrained=True)
    
    # Phase 1: Contrastive learning
    model, contrastive_loss_history = train_contrastive(model, train_loader, cfg, device)
    
    # Phase 2: Linear probe
    model, probe, probe_history, best_val_acc = train_linear_probe(
        model, train_loader, val_loader, cfg, device, 
        cfg["checkpoint_dir"], best_model_path
    )
    
    # Combine history
    history = {
        "contrastive_loss": contrastive_loss_history,
        "probe_train_loss": probe_history["train_loss"],
        "probe_train_acc": probe_history["train_acc"],
        "probe_val_loss": probe_history["val_loss"],
        "probe_val_acc": probe_history["val_acc"],
        "best_val_acc": best_val_acc
    }
    
    # Save training curves
    save_training_curves(history, cfg["curve_save_path"])
    
    print("TRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_acc*100:.1f}%")
    print(f"Model saved to: {best_model_path}")
    print(f"Training curves saved to: {cfg['curve_save_path']}")
    
    return model, history



def load_simclr_model(model_path, device="cpu"):
    
    from Models.simclr import get_simclr_model, get_linear_probe
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = get_simclr_model(projection_dim=128, pretrained=False)
    probe = get_linear_probe(input_dim=512, num_classes=NUM_CLASSES)
    
    model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
    probe.load_state_dict(checkpoint["probe_state_dict"])
    
    model.eval()
    probe.eval()
    
    return model, probe


if __name__ == "__main__":
    print("SimCLR Training for Banknote Recognition")
    print(f"Project root : {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    print(f"Dataset path : {CONFIG['dataset_root']}\n")
    
    # Load dataset
    dataset = BanknoteDataset(
        root=CONFIG["dataset_root"],
        augment=True,
        colour=True
    )
    
    # Train
    model, history = train(dataset)
    
    print("\n Training script completed successfully!")
    print(f"   Best accuracy: {history['best_val_acc']*100:.1f}%")
    print(f"   Model: checkpoints/{CONFIG['best_model_name']}")
    print(f"   Curves: {CONFIG['curve_save_path']}")