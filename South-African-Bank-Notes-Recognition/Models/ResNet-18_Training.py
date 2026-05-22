import os, sys, copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

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
CLASS_LABELS      = _mod.CLASS_LABELS
NUM_CLASSES       = _mod.NUM_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CONFIG = {
    "dataset_root"   : os.path.join(PROJECT_ROOT, "Dataset", "raw",
                                    "Banknote_Dataset_(2005-2023)"),
    "val_split"      : 0.3,
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
    "best_model_name": "resnet18_best.pth",
    "curve_save_path": os.path.join(PROJECT_ROOT, "training_curves_ResNet.png"),
}


def normalise_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for image_tensors, label_tensors in loader:
        image_tensors = torch.stack([normalise_tensor(t) for t in image_tensors])
        image_tensors = image_tensors.to(device)
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

    return (running_loss / total if total else 0.0,
            correct / total      if total else 0.0)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for image_tensors, label_tensors in loader:
            image_tensors = torch.stack([normalise_tensor(t) for t in image_tensors])
            image_tensors = image_tensors.to(device)
            label_tensors = label_tensors.to(device)

            outputs = model(image_tensors)
            loss    = criterion(outputs, label_tensors)

            running_loss += loss.item() * image_tensors.size(0)
            _, predicted  = torch.max(outputs, 1)
            correct      += (predicted == label_tensors).sum().item()
            total        += image_tensors.size(0)

    return (running_loss / total if total else 0.0,
            correct / total      if total else 0.0)


def save_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train", marker="o")
    ax1.plot(epochs, history["val_loss"],   label="Val",   marker="s")
    ax1.set_title("Cross-Entropy Loss (label smoothing=0.1)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, [a*100 for a in history["train_acc"]], label="Train", marker="o")
    ax2.plot(epochs, [a*100 for a in history["val_acc"]],   label="Val",   marker="s")
    ax2.set_title("Classification Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved -> {save_path}")


def train(dataset, config: dict = None):
    cfg    = {**CONFIG, **(config or {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on  : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    n_total = len(dataset)
    n_val   = max(1, int(n_total * cfg["val_split"]))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"])
    val_loader   = DataLoader(val_set,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"])

    print(f"Train samples: {n_train}  |  Val samples: {n_val}\n")

    model = build_resnet18(pretrained=cfg["pretrained"], freeze_backbone=True)
    model.to(device)

    # Label smoothing loss - prevents the model from becoming overconfident on wrong predictions, improving generalisation on unseen images
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["label_smoothing"]
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_path     = os.path.join(cfg["checkpoint_dir"], cfg["best_model_name"])
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_weights  = copy.deepcopy(model.state_dict())
    history       = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}


    # PHASE 1 - head warmup
    print(f"Phase 1: Head warmup ({cfg['freeze_epochs']} epochs) ")
    optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr_head"], weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["freeze_epochs"], eta_min=1e-5
    )

    for epoch in range(1, cfg["freeze_epochs"] + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimiser, device)
        vl, va = validate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        print(f"Epoch {epoch:02d}/{cfg['freeze_epochs']:02d} | "
              f"Train Loss: {tl:.4f}  Acc: {ta*100:.1f}% | "
              f"Val Loss: {vl:.4f}  Acc: {va*100:.1f}%")
        # Save if: higher accuracy, OR same accuracy with lower loss
        if va > best_val_acc or (va == best_val_acc and vl < best_val_loss):
            best_val_acc = va; best_val_loss = vl
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_weights,
                        "val_acc": best_val_acc,
                        "val_loss": best_val_loss,
                        "epoch": epoch}, best_path)
            print(f"  -> Best model saved (val_acc={best_val_acc*100:.1f}%  val_loss={best_val_loss:.4f})")


    # PHASE 2 - full fine-tuning with cosine annealing
    remaining = cfg["total_epochs"] - cfg["freeze_epochs"]
    if remaining > 0:
        print(f"\n Phase 2: Full fine-tuning ({remaining} epochs) ")
        model     = unfreeze_backbone(model)
        optimiser = optim.Adam(
            model.parameters(),
            lr=cfg["lr_finetune"], weight_decay=cfg["weight_decay"]
        )
        # CosineAnnealingLR smoothly reduces LR from lr_finetune → eta_min
        # This avoids the sudden LR drops of StepLR and finds sharper minima
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=remaining, eta_min=1e-6
        )

        for epoch in range(cfg["freeze_epochs"] + 1, cfg["total_epochs"] + 1):
            tl, ta = train_one_epoch(model, train_loader, criterion, optimiser, device)
            vl, va = validate(model, val_loader, criterion, device)
            scheduler.step()
            history["train_loss"].append(tl); history["val_loss"].append(vl)
            history["train_acc"].append(ta);  history["val_acc"].append(va)
            print(f"Epoch {epoch:02d}/{cfg['total_epochs']:02d} | "
                  f"Train Loss: {tl:.4f}  Acc: {ta*100:.1f}% | "
                  f"Val Loss: {vl:.4f}  Acc: {va*100:.1f}%")
            # Save if: higher accuracy, OR same accuracy with lower loss
            if va > best_val_acc or (va == best_val_acc and vl < best_val_loss):
                best_val_acc = va; best_val_loss = vl
                best_weights = copy.deepcopy(model.state_dict())
                torch.save({"model_state_dict": best_weights,
                            "val_acc": best_val_acc,
                            "val_loss": best_val_loss,
                            "epoch": epoch}, best_path)
                print(f"  -> Best model saved (val_acc={best_val_acc*100:.1f}%  val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.1f}%")
    print(f"Model saved to: {best_path}")
    model.load_state_dict(best_weights)
    save_training_curves(history, cfg["curve_save_path"])
    return model, history


if __name__ == "__main__":
    from Data.dataloader import BanknoteDataset
    print("ResNet-18")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset path : {CONFIG['dataset_root']}\n")
    dataset = BanknoteDataset(root=CONFIG["dataset_root"],
                              augment=True, colour=True)
    model, history = train(dataset)