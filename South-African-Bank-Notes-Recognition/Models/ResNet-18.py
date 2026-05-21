import torch
import torch.nn as nn
from torchvision import models

# Label mapping — must match dataloader output
CLASS_LABELS = {0: "R10", 1: "R20", 2: "R50", 3: "R100", 4: "R200"}
NUM_CLASSES  = 5


def build_resnet18(pretrained: bool = True,
                   freeze_backbone: bool = True,
                   dropout_rate: float = 0.4) -> nn.Module:
    """
    ResNet-18 adapted for South African banknote classification.

    Improvements over baseline:
        - Dropout (p=0.4) before the FC head reduces overfitting on the
          small 68-image dataset.  During training this randomly zeros 40%
          of the 512 features, forcing the head to learn redundant
          representations rather than memorising specific training images.
        - Two-layer head (512→256→5) gives the classifier more capacity to
          learn denomination-specific boundaries in the feature space.

    Args:
        pretrained      : Load ImageNet weights (recommended).
        freeze_backbone : Freeze all layers except head for Phase 1 training.
        dropout_rate    : Dropout probability before the FC head (default 0.4).
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace FC head with Dropout + two-layer classifier
    in_features = model.fc.in_features   # 512
    model.fc    = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate / 2),
        nn.Linear(256, NUM_CLASSES),
    )

    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def predict(model: nn.Module,
            image_tensor: torch.Tensor,
            device: torch.device) -> dict:
    """
    Run inference on a single preprocessed banknote image tensor.
    Designed to be called directly by the GUI.

    Args:
        model        : Trained ResNet-18.
        image_tensor : Shape (3, 224, 224), float32, values in [0, 1].
        device       : torch.device.

    Returns:
        dict with 'label', 'class_index', 'confidence', 'probabilities'.
    """
    model.eval()
    model.to(device)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits        = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_index = predicted_index.item()
    confidence      = confidence.item()

    return {
        "label"        : CLASS_LABELS[predicted_index],
        "class_index"  : predicted_index,
        "confidence"   : round(confidence, 4),
        "probabilities": {
            CLASS_LABELS[i]: round(probabilities[0, i].item(), 4)
            for i in range(NUM_CLASSES)
        },
    }


def load_model(checkpoint_path: str,
               device: torch.device,
               freeze_backbone: bool = False) -> nn.Module:
    """Load a saved checkpoint for inference or continued training."""
    model      = build_resnet18(pretrained=False,
                                freeze_backbone=freeze_backbone)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_resnet18(pretrained=True, freeze_backbone=True)
    model.to(device)

    dummy = torch.randn(1, 3, 224, 224).to(device)
    result = predict(model, dummy, device)
    print("Prediction:", result)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")