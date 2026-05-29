import sys
import time
import os
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports
from Data.preprocessing import enhance_resolution
from Data.segmentation import segment_note
from Models.SimCLR import get_simclr_model, get_linear_probe
from Models.SIFT_FLANN import SIFTFLANNClassifier

# Model imports
import importlib.util
_spec = importlib.util.spec_from_file_location("ResNet18", os.path.join(PROJECT_ROOT, "Models", "ResNet-18.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_resnet18 = _mod.build_resnet18
CLASS_LABELS = _mod.CLASS_LABELS

# Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Model Loading

def load_resnet_model():
    path = os.path.join(PROJECT_ROOT, "checkpoints", "resnet18_best.pth")
    checkpoint = torch.load(path, map_location=DEVICE)
    model = build_resnet18(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    return model


def load_simclr_model():
    path = os.path.join(PROJECT_ROOT, "checkpoints", "simclr_best.pth")
    checkpoint = torch.load(path, map_location=DEVICE)
    model = get_simclr_model(projection_dim=128, pretrained=False)
    probe = get_linear_probe(input_dim=512, num_classes=5)
    model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    probe.load_state_dict(checkpoint['probe_state_dict'])
    model.to(DEVICE).eval()
    probe.to(DEVICE).eval()
    return model, probe


def load_sift_classifier():
    # Load from the pre-segmented dataset folder for max consistency
    ref_dir = os.path.join(PROJECT_ROOT, "Dataset", "segmented")
    clf = SIFTFLANNClassifier(ratio_threshold=0.75, min_match_count=15, inlier_threshold=0.35)
    for f in os.listdir(ref_dir):
        if f.endswith(('.png', '.jpg')):
            denom = f.split('_')[0].upper()
            if denom in CLASS_LABELS:
                img = cv2.imread(os.path.join(ref_dir, f), cv2.IMREAD_GRAYSCALE)
                clf.fit(img, denom)
    return clf


def prepare_for_dl(image_rgb):
    resized = cv2.resize(image_rgb, (224, 224))
    img_norm = ((resized.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

def ensemble_predict(r_res, s_res, f_res):
    # Initialize scores for all valid CLASS_LABELS
    scores = {label: 0.0 for label in CLASS_LABELS}
    
    # Process ResNet
    if r_res and r_res[0] in scores and r_res[1] > 20: 
        scores[r_res[0]] += (r_res[1] / 100.0)
    
    # Process SimCLR
    if s_res and s_res[0] in scores and s_res[1] > 20: 
        scores[s_res[0]] += (s_res[1] / 100.0)
    
    # Process SIFT (Check if label exists in our defined classes)
    if f_res and f_res[0] in scores: 
        scores[f_res[0]] += (f_res[1] / 100.0) * 0.3
    
    # Find the winner
    winner = max(scores, key=scores.get)
    
    # Return Unknown if max score is below threshold
    if scores[winner] <=0:
        return "Unknown"
        
    return winner

def get_predictions(model_resnet, model_simclr, probe_simclr, clf_sift, image_bgr):
  
    # 1. Segment 
    seg_bgr = segment_note(image_bgr)
    if seg_bgr is None: return None, None, None
    
    # 2. Enhance 
    seg_bgr = enhance_resolution(seg_bgr, target_width=640)
    
    # 3. Convert to RGB for model input
    seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
    
    # 4. Prepare for DL 
    dl_input = prepare_for_dl(seg_rgb)
    
    # Model Inference
    
    # ResNet
    with torch.no_grad():
        out = model_resnet(dl_input)
        prob = torch.nn.functional.softmax(out, dim=1)
        r_conf, r_idx = torch.max(prob, 1)
        r_res = (CLASS_LABELS[r_idx.item()], r_conf.item() * 100)
        
    # SimCLR
    with torch.no_grad():
        feat = model_simclr.get_features(dl_input) 
        out = probe_simclr(feat)
        prob = torch.nn.functional.softmax(out, dim=1)
        s_conf, s_idx = torch.max(prob, 1)
        s_res = (CLASS_LABELS[s_idx.item()], s_conf.item() * 100)
        
    # SIFT 
    sift_gray = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2GRAY)
    f_label, f_conf, _ = clf_sift.predict(sift_gray)
    f_res = (f_label, f_conf * 100)
    
    return r_res, s_res, f_res

def run_evaluation(test_dir):
    # Initialize Models
    resnet = load_resnet_model()
    simclr, probe = load_simclr_model()
    sift = load_sift_classifier()
    
    results = {"ResNet": 0, "SimCLR": 0, "SIFT": 0, "Ensemble": 0, "Total": 0}
    
    files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print(f"ERROR: No images found in {test_dir}")
        return

    for f in files:
        # 1. Extract the True Label from the filename 
        true_label = f.split('_')[0].upper()
        
        # Skip if the label isn't in defined CLASS_LABELS
        if true_label not in CLASS_LABELS:
            print(f"Skipping {f}: label '{true_label}' not in CLASS_LABELS")
            continue

        img_path = os.path.join(test_dir, f)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 2. Predict
        r_res, s_res, f_res = get_predictions(resnet, simclr, probe, sift, img)
        if r_res is None: continue
        
        # 3. Tally
        results["Total"] += 1
        if r_res[0] == true_label: results["ResNet"] += 1
        if s_res[0] == true_label: results["SimCLR"] += 1
        if f_res and f_res[0] == true_label: results["SIFT"] += 1
        
        ensemble_choice = ensemble_predict(r_res, s_res, f_res)
        if ensemble_choice == true_label: results["Ensemble"] += 1

    # Report
    if results["Total"] > 0:
        print("--- Results ---")
        print(f'Number of Images in test set: {len(files)}')
        for model, count in results.items():
            if model == "Total": continue
            acc = (count / results["Total"]) * 100
            print(f"{model} Accuracy: {acc:.2f}% ({count}/{results['Total']})")
    else:
        print("Evaluation finished with 0 images processed.")

if __name__ == "__main__":
    run_evaluation("Dataset/test") 