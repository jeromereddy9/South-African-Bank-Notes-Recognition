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
from Data.preprocessing import preprocessing_CLAHE
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

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(page_title="SA Banknote Recognition", page_icon="💰", layout="wide")

# ============================================
# MODEL LOADING FUNCTIONS
# ============================================
@st.cache_resource
def load_resnet_model():
    path = os.path.join(PROJECT_ROOT, "checkpoints", "resnet18_best.pth")
    checkpoint = torch.load(path, map_location=DEVICE)
    model = build_resnet18(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    return model

@st.cache_resource
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

@st.cache_resource
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

# ============================================
# PIPELINE & CLASSIFICATION
# ============================================
def prepare_for_dl(image_rgb):
    resized = cv2.resize(image_rgb, (224, 224))
    img_norm = ((resized.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

def ensemble_predict(r_res, s_res, f_res):
    scores = {label: 0.0 for label in CLASS_LABELS}
    if r_res and r_res[1] > 20: scores[r_res[0]] += (r_res[1] / 100.0)
    if s_res and s_res[1] > 20: scores[s_res[0]] += (s_res[1] / 100.0)
    if f_res and f_res[0] != "Unknown": scores[f_res[0]] += (f_res[1] / 100.0) * 0.3
    
    winner = max(scores, key=scores.get)
    return winner if scores[winner] >= 0.5 else "Unknown"

# ============================================
# MAIN GUI
# ============================================
st.title("💰 SA Banknote Recognition")
resnet, (simclr, probe), sift = load_resnet_model(), load_simclr_model(), load_sift_classifier()

uploaded_file = st.file_uploader("Upload Banknote", type=["png", "jpg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_array = np.array(image)
    
    if st.button("Run Pipeline"):
        # 1. Segment
        seg_bgr = segment_note(cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB) if seg_bgr is not None else image_array
        st.image(seg_rgb, caption="Extracted Banknote")

        # 2. Predict
        tensor = prepare_for_dl(seg_rgb)
        with torch.no_grad():
            r_idx = torch.argmax(resnet(tensor))
            s_idx = torch.argmax(probe(simclr.get_features(tensor)))
        
        r_pred = (CLASS_LABELS[r_idx.item()], 95.0) # Simplified for display
        s_pred = (CLASS_LABELS[s_idx.item()], 90.0)
        f_label, f_conf, _ = sift.predict(cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2GRAY))
        f_pred = (f_label, f_conf * 100)
        
        final = ensemble_predict(r_pred, s_pred, f_pred)
        st.success(f"### Prediction: {final}")