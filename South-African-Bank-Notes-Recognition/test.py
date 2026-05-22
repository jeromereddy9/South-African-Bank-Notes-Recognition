"""
quick_model_analysis.py
Quick analysis of all three models on raw dataset images.
"""

import sys
import os
import torch
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Data.preprocessing import preprocessing_global
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ResNet18", os.path.join(PROJECT_ROOT, "Models", "ResNet-18.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_resnet18    = _mod.build_resnet18
CLASS_LABELS      = _mod.CLASS_LABELS
from Models.SimCLR import get_simclr_model, get_linear_probe
from Models.SIFT_FLANN import SIFTFLANNClassifier

CLASS_TO_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}
INDEX_TO_CLASS = {i: label for i, label in enumerate(CLASS_LABELS)}


def find_dataset_path():
    """Find the dataset path by checking common locations."""
    possible_paths = [
        "Dataset/raw/Banknote_Dataset_(2005-2023)",
        os.path.join(PROJECT_ROOT, "Dataset", "raw", "Banknote_Dataset_(2005-2023)"),
        "../Dataset/raw/Banknote_Dataset_(2005-2023)",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def preprocess_image_raw(image_path):
    """Load and preprocess image (no augmentation)."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Normalize to [0, 1]
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Grayscale for SIFT
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    return tensor, gray


def load_models():
    """Load all three models."""
    models = {}
    
    # Find dataset path
    dataset_path = find_dataset_path()
    if dataset_path is None:
        print("❌ Could not find dataset path!")
        return models
    
    print(f"📁 Dataset path: {dataset_path}")
    
    # Load ResNet
    print("Loading ResNet...")
    try:
        checkpoint = torch.load("checkpoints/resnet18_best.pth", map_location='cpu')
        model = build_resnet18(pretrained=False, freeze_backbone=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models['resnet'] = model
        print(f"  ✅ ResNet loaded")
    except Exception as e:
        print(f"  ❌ ResNet failed: {e}")
        models['resnet'] = None
    
    # Load SimCLR
    print("Loading SimCLR...")
    try:
        checkpoint = torch.load("checkpoints/simclr_best.pth", map_location='cpu')
        model = get_simclr_model(projection_dim=128, pretrained=False)
        probe = get_linear_probe(input_dim=512, num_classes=5)
        model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        probe.load_state_dict(checkpoint['probe_state_dict'])
        model.eval()
        probe.eval()
        models['simclr'] = (model, probe)
        print(f"  ✅ SimCLR loaded")
    except Exception as e:
        print(f"  ❌ SimCLR failed: {e}")
        models['simclr'] = None
    
    # Load SIFT (builds references on the fly)
    print("Loading SIFT (building references)...")
    try:
        classifier = SIFTFLANNClassifier(
            ratio_threshold=0.75,
            min_match_count=15,
            inlier_threshold=0.35
        )
        
        print(f"   Initial classifier size: {len(classifier)}")
        
        # Build references from dataset
        files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
        print(f"   Found {len(files)} reference images")
        
        count = 0
        for f in files:
            parts = f.split('_')
            if len(parts) >= 1:
                denom = parts[0].upper()
                if denom in CLASS_LABELS:
                    img_path = os.path.join(dataset_path, f)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = preprocessing_global(img)
                        img = cv2.resize(img, (224, 224))
                        classifier.fit(img, denom)
                        count += 1
                        if count % 10 == 0:
                            print(f"     Fitted {count} images, DB size: {len(classifier)}")
        
        models['sift'] = classifier
        print(f"  ✅ SIFT loaded ({len(classifier)} references)")
    except Exception as e:
        print(f"  ❌ SIFT failed: {e}")
        models['sift'] = None
    
    return models


def test_model_on_image(model, image_tensor, model_type, simclr_probe=None):
    """Test a single image on a model."""
    with torch.no_grad():
        if model_type == 'resnet':
            outputs = model(image_tensor)
        elif model_type == 'simclr':
            features = model.get_features(image_tensor)
            outputs = simclr_probe(features)
        else:
            return None, None
    
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item()


def main():
    print("=" * 60)
    print("QUICK MODEL ANALYSIS")
    print("=" * 60)
    
    # Load models
    models = load_models()
    
    if not models:
        print("No models loaded!")
        return
    
    # Get test images
    dataset_path = find_dataset_path()
    if dataset_path is None:
        print("❌ Could not find dataset path!")
        return
    
    test_images = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
    
    print(f"\n📁 Testing on {len(test_images)} images\n")
    
    # Results storage
    results = {
        'resnet': {'correct': 0, 'total': 0, 'details': []},
        'simclr': {'correct': 0, 'total': 0, 'details': []},
        'sift': {'correct': 0, 'total': 0, 'details': []}
    }
    
    for idx, filename in enumerate(test_images):
        # Parse filename
        parts = filename.split('_')
        if len(parts) < 4:
            continue
        
        denom = parts[0].upper()
        if denom not in CLASS_LABELS:
            continue
        
        true_label = CLASS_TO_INDEX[denom]
        img_path = os.path.join(dataset_path, filename)
        
        # Preprocess
        tensor, gray = preprocess_image_raw(img_path)
        if tensor is None:
            continue
        
        # Test ResNet
        if models.get('resnet'):
            pred, conf = test_model_on_image(models['resnet'], tensor, 'resnet')
            is_correct = (pred == true_label)
            results['resnet']['total'] += 1
            if is_correct:
                results['resnet']['correct'] += 1
            results['resnet']['details'].append({
                'file': filename, 'true': denom, 'pred': INDEX_TO_CLASS[pred], 'conf': conf
            })
        
        # Test SimCLR
        if models.get('simclr'):
            model, probe = models['simclr']
            pred, conf = test_model_on_image(model, tensor, 'simclr', probe)
            is_correct = (pred == true_label)
            results['simclr']['total'] += 1
            if is_correct:
                results['simclr']['correct'] += 1
            results['simclr']['details'].append({
                'file': filename, 'true': denom, 'pred': INDEX_TO_CLASS[pred], 'conf': conf
            })
        
        # Test SIFT
        if models.get('sift') and len(models['sift']) > 0:
            label, confidence, _ = models['sift'].predict(gray)
            if label == "unknown":
                pred = "Unknown"
                is_correct = False
            else:
                pred = label
                is_correct = (label == denom)
            results['sift']['total'] += 1
            if is_correct:
                results['sift']['correct'] += 1
            results['sift']['details'].append({
                'file': filename, 'true': denom, 'pred': pred, 'conf': confidence
            })
        
        # Print progress every 10 images
        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(test_images)} images")
    
    # Print detailed results per image
    print("\n" + "=" * 60)
    print("PER-IMAGE RESULTS (first 20)")
    print("=" * 60)
    
    for i, detail in enumerate(results['resnet']['details'][:20]):
        print(f"\n{i+1}. {detail['file']}")
        print(f"   True: {detail['true']}")
        
        # Find corresponding predictions
        resnet_detail = next((d for d in results['resnet']['details'] if d['file'] == detail['file']), None)
        simclr_detail = next((d for d in results['simclr']['details'] if d['file'] == detail['file']), None)
        sift_detail = next((d for d in results['sift']['details'] if d['file'] == detail['file']), None)
        
        if resnet_detail:
            correct = "✅" if resnet_detail['true'] == resnet_detail['pred'] else "❌"
            print(f"   ResNet: {resnet_detail['pred']} ({resnet_detail['conf']*100:.1f}%) {correct}")
        
        if simclr_detail:
            correct = "✅" if simclr_detail['true'] == simclr_detail['pred'] else "❌"
            print(f"   SimCLR: {simclr_detail['pred']} ({simclr_detail['conf']*100:.1f}%) {correct}")
        
        if sift_detail:
            correct = "✅" if sift_detail['true'] == sift_detail['pred'] else "❌"
            print(f"   SIFT: {sift_detail['pred']} ({sift_detail['conf']*100:.1f}%) {correct}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model_name in ['resnet', 'simclr', 'sift']:
        r = results[model_name]
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"\n{model_name.upper()}:")
            print(f"   Accuracy: {acc:.1f}% ({r['correct']}/{r['total']})")
            
            # Show misclassifications
            mis = [d for d in r['details'] if d['true'] != d['pred']]
            if mis:
                print(f"   Misclassifications ({len(mis)}):")
                for m in mis[:10]:
                    print(f"     {m['file']}: True={m['true']}, Pred={m['pred']} ({m['conf']*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()