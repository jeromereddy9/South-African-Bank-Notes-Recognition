import os, sys, cv2, torch, numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Setup Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports
from Models.SimCLR import get_simclr_model, get_linear_probe
from Models.SIFT_FLANN import SIFTFLANNClassifier
from Data.augmentation import generate_enhanced_references

import importlib.util
resnet_spec = importlib.util.spec_from_file_location("ResNet18", os.path.join(PROJECT_ROOT, "Models", "ResNet-18.py"))
resnet_mod = importlib.util.module_from_spec(resnet_spec)
resnet_spec.loader.exec_module(resnet_mod)
build_resnet18 = resnet_mod.build_resnet18

# Configuration
TEST_DIR = os.path.join(PROJECT_ROOT, "Dataset", "test")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
REF_DIR = os.path.join(PROJECT_ROOT, "Dataset", "segmented") 
CLASS_LABELS = ["R10", "R20", "R50", "R100", "R200"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- LOADING FUNCTIONS ---
def load_resnet_eval():
    model = build_resnet18(pretrained=False)
    path = os.path.join(CHECKPOINT_DIR, "resnet18_best.pth")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
    return model.to(device).eval()

def load_simclr_eval():
    model = get_simclr_model(projection_dim=128, pretrained=False)
    probe = get_linear_probe(input_dim=512, num_classes=5)
    path = os.path.join(CHECKPOINT_DIR, "simclr_best.pth")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.backbone.load_state_dict(ckpt['backbone_state_dict'])
        probe.load_state_dict(ckpt['probe_state_dict'])
    return model.to(device).eval(), probe.to(device).eval()

def load_sift_eval():
    clf = SIFTFLANNClassifier(ratio_threshold=0.75, min_match_count=15, inlier_threshold=0.35)
    for f in os.listdir(REF_DIR):
        if f.endswith(('.png', '.jpg')):
            denom = f.split('_')[0].upper()
            if denom in CLASS_LABELS:
                img = cv2.imread(os.path.join(REF_DIR, f), cv2.IMREAD_GRAYSCALE)
                for ref in generate_enhanced_references(img, num_variations=5):
                    clf.fit(ref, denom)
    return clf

# --- MAIN EVALUATION ---
def main():
    print(f"\n{'='*75}\n{'EVALUATING ON PRE-SEGMENTED DATA':^75}\n{'='*75}")
    resnet, (simclr, probe), sift = load_resnet_eval(), load_simclr_eval(), load_sift_eval()
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg'))])
    
    stats = {lbl: {"total": 0, "r": 0, "s": 0, "f": 0, "e": 0} for lbl in CLASS_LABELS}
    
    print(f"{'File':<15} | {'True':<5} | {'Res':<6} | {'Sim':<6} | {'SIFT':<6} | {'Ens':<6}")
    print("-" * 65)

    for f in test_files:
        true = f.split('_')[0].upper()
        img_bgr = cv2.imread(os.path.join(TEST_DIR, f))
        if img_bgr is None: continue
        
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        tensor = transform_pipeline(Image.fromarray(rgb)).unsqueeze(0).to(device)
        
        # DL Inference
        with torch.no_grad():
            r_out = torch.softmax(resnet(tensor), dim=1)
            r_conf, r_idx = torch.max(r_out, dim=1)
            
            s_out = torch.softmax(probe(simclr.get_features(tensor)), dim=1)
            s_conf, s_idx = torch.max(s_out, dim=1)
            
        r_pred, s_pred = CLASS_LABELS[r_idx], CLASS_LABELS[s_idx]
        
        # SIFT Inference
        f_label, f_conf, _ = sift.predict(gray)
        f_pred = f_label if f_label in CLASS_LABELS else None
        
        # Weighted Ensemble with Unknown Check
        scores = {label: 0.0 for label in CLASS_LABELS}
        if r_conf.item() > 0.2: scores[r_pred] += r_conf.item()
        if s_conf.item() > 0.2: scores[s_pred] += s_conf.item()
        if f_pred: scores[f_pred] += (f_conf * 0.3)
        
        ens = max(scores, key=scores.get)
        if max(scores.values()) < 0.5: ens = "Unknown"
        
        # Stats
        stats[true]["total"] += 1
        if r_pred == true: stats[true]["r"] += 1
        if s_pred == true: stats[true]["s"] += 1
        if f_pred == true: stats[true]["f"] += 1
        if ens == true: stats[true]["e"] += 1
        
        print(f"{f:<15} | {true:<5} | {r_pred:<6} | {s_pred:<6} | {f_pred or 'None':<6} | {ens:<6}")

    print("\nFinal Accuracy Report:")
    for l in CLASS_LABELS:
        t = stats[l]["total"]
        if t > 0: print(f"{l}: Res={stats[l]['r']/t:.1%} | Sim={stats[l]['s']/t:.1%} | SIFT={stats[l]['f']/t:.1%} | Ens={stats[l]['e']/t:.1%}")

if __name__ == "__main__":
    main()