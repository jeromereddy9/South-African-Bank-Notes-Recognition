"""
ResNet-18 Model for Banknote Classification
Corrected implementation with proper CLASS_LABELS structure
"""

import torch
import torch.nn as nn
from torchvision import models

# Label mapping (corrected to list for consistency)
CLASS_LABELS = ["R10", "R20", "R50", "R100", "R200"]
NUM_CLASSES = len(CLASS_LABELS)


def build_resnet18(pretrained: bool = True,
                   freeze_backbone: bool = True,
                   dropout_rate: float = 0.4) -> nn.Module:
    """
    Build ResNet-18 model for banknote classification.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone layers (transfer learning)
        dropout_rate: Dropout rate for regularization
    
    Returns:
        ResNet-18 model with custom classifier head
    """
    # Load pretrained or random weights
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Freeze backbone if doing transfer learning
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace FC head with Dropout + two-layer classifier
    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate / 2),
        nn.Linear(256, NUM_CLASSES),
    )
    
    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """
    Unfreeze all parameters for full fine-tuning.
    
    Args:
        model: ResNet-18 model
    
    Returns:
        Model with all parameters unfrozen
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


def predict(model: nn.Module,
            image_tensor: torch.Tensor,
            device: torch.device) -> dict:
    """
    Run inference on a single preprocessed banknote image tensor.
    
    Args:
        model: Trained ResNet-18 model
        image_tensor: Shape (3, 224, 224), float32, values in [0, 1]
        device: torch.device for computation
    
    Returns:
        dict with 'label', 'class_index', 'confidence', 'probabilities'
    """
    model.eval()
    model.to(device)
    
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)
    
    predicted_index = predicted_index.item()
    confidence = confidence.item()
    
    return {
        "label": CLASS_LABELS[predicted_index],
        "class_index": predicted_index,
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_LABELS[i]: round(probabilities[0, i].item(), 4)
            for i in range(NUM_CLASSES)
        },
    }


def load_model(checkpoint_path: str,
               device: torch.device,
               freeze_backbone: bool = False) -> nn.Module:
    """
    Load a saved checkpoint for inference or continued training.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        device: torch.device for model
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        Loaded model ready for inference
    """
    model = build_resnet18(pretrained=False, freeze_backbone=freeze_backbone)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build model
    model = build_resnet18(pretrained=True, freeze_backbone=True)
    model.to(device)
    
    # Test with dummy input
    dummy = torch.randn(1, 3, 224, 224).to(device)
    result = predict(model, dummy, device)
    
    print("\nPrediction:", result)
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable:,} / Total: {total:,}")
    
    print(f"\nClass labels: {CLASS_LABELS}")