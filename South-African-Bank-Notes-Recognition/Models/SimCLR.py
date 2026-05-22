"""
SimCLR (Simple Framework for Contrastive Learning of Visual Representations)
Corrected implementation with proper NT-Xent loss and channel handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimCLR(nn.Module):
    """
    SimCLR model with ResNet-18 backbone and projection head.
    """
    
    def __init__(self, projection_dim=128, pretrained=True, input_channels=3):
        """
        Args:
            projection_dim: Dimension of projection head output
            pretrained: Use ImageNet pretrained weights
            input_channels: 3 for RGB, 1 for grayscale
        """
        super(SimCLR, self).__init__()
        
        # Load ResNet-18 backbone
        self.backbone = models.resnet18(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        # Modify first conv layer for different input channels if needed
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Get feature dimension (512 for ResNet-18)
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace final FC with identity to get features
        self.backbone.fc = nn.Identity()
        
        # Projection head (maps features to embedding space)
        # Standard SimCLR uses: Linear -> ReLU -> Linear
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, projection_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through backbone and projection head.
        
        Args:
            x: Input images (batch_size, channels, height, width)
        
        Returns:
            Normalized embeddings (batch_size, projection_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (batch_size, 512)
        
        # Project to embedding space
        embeddings = self.projection_head(features)  # (batch_size, projection_dim)
        
        # Normalize embeddings to unit sphere (cosine similarity)
        embeddings = F.normalize(embeddings, dim=1)
        
        return embeddings
    
    def get_features(self, x):
        """
        Extract backbone features without projection (for linear probe).
        
        Args:
            x: Input images (batch_size, channels, height, width)
        
        Returns:
            Backbone features (batch_size, 512)
        """
        return self.backbone(x)


class LinearProbe(nn.Module):
    """
    Linear classifier for evaluating learned representations.
    """
    
    def __init__(self, input_dim=512, num_classes=5):
        """
        Args:
            input_dim: Dimension of input features (512 for ResNet-18)
            num_classes: Number of output classes
        """
        super(LinearProbe, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, features):
        """
        Args:
            features: Backbone features (batch_size, input_dim)
        
        Returns:
            Logits (batch_size, num_classes)
        """
        return self.classifier(features)


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
    Also known as InfoNCE loss.
    
    This is the standard SimCLR loss function.
    """
    
    def __init__(self, temperature=0.5):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings1, embeddings2):
        """
        Compute contrastive loss for a batch of positive pairs.
        
        Args:
            embeddings1: First augmentation embeddings (batch_size, projection_dim)
            embeddings2: Second augmentation embeddings (batch_size, projection_dim)
        
        Returns:
            Scalar loss value
        """
        batch_size = embeddings1.shape[0]
        device = embeddings1.device
        
        # Concatenate both augmentations
        # embeddings shape: (2 * batch_size, projection_dim)
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)
        
        # Compute cosine similarity matrix
        # similarity shape: (2 * batch_size, 2 * batch_size)
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive pair labels
        # For each sample i in first half, its positive is at position i + batch_size
        # For each sample i in second half, its positive is at position i - batch_size
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(device)
        
        # Mask out self-similarity (diagonal elements)
        # These should not contribute to loss
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        similarity = similarity.masked_fill(mask, -1e9)
        
        # Compute cross-entropy loss
        # This treats it as a (2*batch_size)-way classification problem
        # where each sample must identify its positive pair
        loss = F.cross_entropy(similarity, labels)
        
        return loss


def get_simclr_model(projection_dim=128, pretrained=True, input_channels=3):
    """
    Factory function to create SimCLR model.
    
    Args:
        projection_dim: Dimension of projection space
        pretrained: Use ImageNet pretrained weights
        input_channels: 3 for RGB, 1 for grayscale
    
    Returns:
        SimCLR model
    """
    return SimCLR(
        projection_dim=projection_dim,
        pretrained=pretrained,
        input_channels=input_channels
    )


def get_linear_probe(input_dim=512, num_classes=5):
    """
    Factory function to create linear probe classifier.
    
    Args:
        input_dim: Dimension of backbone features (512 for ResNet-18)
        num_classes: Number of output classes
    
    Returns:
        LinearProbe model
    """
    return LinearProbe(input_dim=input_dim, num_classes=num_classes)


def predict(simclr_model, linear_probe, image_tensor, device):
    """
    Run inference using SimCLR backbone + linear probe.
    
    Args:
        simclr_model: Trained SimCLR model
        linear_probe: Trained linear probe classifier
        image_tensor: Shape (3, 224, 224), float32, values in [0, 1]
        device: torch.device
    
    Returns:
        dict with 'class_index', 'confidence', 'probabilities'
    """
    simclr_model.eval()
    linear_probe.eval()
    simclr_model.to(device)
    linear_probe.to(device)
    
    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Extract features
        features = simclr_model.get_features(image_tensor)
        
        # Classify
        logits = linear_probe(features)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)
    
    predicted_index = predicted_index.item()
    confidence = confidence.item()
    
    CLASS_LABELS = ["R10", "R20", "R50", "R100", "R200"]
    
    return {
        "label": CLASS_LABELS[predicted_index],
        "class_index": predicted_index,
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_LABELS[i]: round(probabilities[0, i].item(), 4)
            for i in range(5)
        },
    }


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model (RGB input)
    model = get_simclr_model(projection_dim=128, pretrained=True, input_channels=3)
    model.to(device)
    
    # Test contrastive learning
    dummy1 = torch.randn(4, 3, 224, 224).to(device)
    dummy2 = torch.randn(4, 3, 224, 224).to(device)
    
    model.eval()
    with torch.no_grad():
        embeddings1 = model(dummy1)
        embeddings2 = model(dummy2)
    
    print(f"\nEmbeddings shape: {embeddings1.shape}")
    print(f"Embeddings are normalized: {torch.allclose(torch.norm(embeddings1, dim=1), torch.ones(4).to(device))}")
    
    # Test loss
    criterion = ContrastiveLoss(temperature=0.5)
    loss = criterion(embeddings1, embeddings2)
    print(f"\nContrastive loss: {loss.item():.4f}")
    
    # Test linear probe
    probe = get_linear_probe(input_dim=512, num_classes=5)
    probe.to(device)
    
    with torch.no_grad():
        features = model.get_features(dummy1)
        logits = probe(features)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
