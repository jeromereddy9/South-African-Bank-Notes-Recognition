"""
South African Banknote Recognition GUI

A simple web-based interface for banknote classification.
Supports individual models or ensemble voting (majority vote across all 3 models).

Run with: streamlit run South_African_Bank_Notes_Recognition.py
"""

import sys
import time
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from collections import Counter

# Import dataloader for preprocessing
from Data.dataloader import BanknoteDataset
from Data.preprocessing import preprocessing_global, preprocessing_CLAHE
from Data.segmentation import segment_note

# Model imports
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

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="SA Banknote Recognition",
    page_icon="💰",
    layout="wide"
)

# ============================================
# MODEL LOADING FUNCTIONS (with caching)
# ============================================

@st.cache_resource
def load_resnet_model(model_path="checkpoints/resnet18_best.pth"):
    """Load trained ResNet-18 model."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model = build_resnet18(pretrained=False, freeze_backbone=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint.get('val_acc', 0.0)
    except Exception as e:
        st.error(f"Failed to load ResNet model: {e}")
        return None, 0.0


@st.cache_resource
def load_simclr_model(model_path="checkpoints/simclr_best.pth"):
    """Load trained SimCLR model with linear probe."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = get_simclr_model(projection_dim=128, pretrained=False)
        probe = get_linear_probe(input_dim=512, num_classes=5)
        
        model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        probe.load_state_dict(checkpoint['probe_state_dict'])
        
        model.eval()
        probe.eval()
        
        return model, probe, checkpoint.get('val_acc', 0.0)
    except Exception as e:
        st.error(f"Failed to load SimCLR model: {e}")
        return None, None, 0.0


@st.cache_resource
def load_sift_classifier(reference_folder="Dataset/raw/Banknote_Dataset_(2005-2023)"):
    """Load SIFT-FLANN classifier with enhanced augmentations for better discrimination."""
    try:
        from Data.preprocessing import preprocessing_global
        from Data.augmentation import generate_enhanced_references
        
        classifier = SIFTFLANNClassifier(
            ratio_threshold=0.75,
            min_match_count=15,
            inlier_threshold=0.35
        )
        
        label_mapping = {'R10': 0, 'R20': 1, 'R50': 2, 'R100': 3, 'R200': 4}
        
        # Count for reporting
        ref_counts = {denom: 0 for denom in label_mapping.keys()}
        
        for filename in os.listdir(reference_folder):
            if filename.endswith('.png'):
                parts = filename.split('_')
                if len(parts) >= 1:
                    denom = parts[0].upper()
                    if denom in label_mapping:
                        img_path = os.path.join(reference_folder, filename)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        
                        img = preprocessing_global(img)
                        img = cv2.resize(img, (224, 224))
                        
                        # Generate enhanced references for this image
                        # Use more variations for R50 (problematic denomination)
                        if denom == "R50":
                            references = generate_enhanced_references(img, denom, num_variations=15)
                        else:
                            references = generate_enhanced_references(img, denom, num_variations=8)
                        
                        for ref in references:
                            classifier.fit(ref, denom)
                            ref_counts[denom] += 1
        
        # Print summary to console
        st.write("SIFT Reference Database Summary:")
        for denom, count in ref_counts.items():
            st.write(f"  {denom}: {count} reference images")
        
        return classifier, sum(ref_counts.values())
        
    except Exception as e:
        st.error(f"Failed to load SIFT classifier: {e}")
        return None, 0


# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(image_array, use_clahe=False):
    """Apply preprocessing matching training pipeline."""
    
    # Ensure image is BGR (OpenCV format)
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    elif image_array.shape[2] == 3:
        # PIL loads as RGB, convert to BGR
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE on V channel (HSV) — matches training
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    hsv = cv2.merge([h, s, v])
    processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Convert to RGB
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Resize
    processed_resized = cv2.resize(processed_rgb, (224, 224))
    
    return processed_resized


def prepare_for_resnet(image_note):
    """Prepare image for ResNet (already RGB, just normalize)."""
    tensor = torch.from_numpy(image_note).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor


def prepare_for_simclr(image_note):
    """Prepare image for SimCLR."""
    return prepare_for_resnet(image_note)


def prepare_for_sift(image_note):
    """Prepare image for SIFT (grayscale uint8)."""
    return cv2.cvtColor(image_note, cv2.COLOR_RGB2GRAY).astype(np.uint8)


# ============================================
# CLASSIFICATION FUNCTIONS
# ============================================

def classify_with_resnet(model, image_note):
    """Run ResNet classification."""
    tensor = prepare_for_resnet(image_note)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    label = CLASS_LABELS[predicted.item()]
    confidence_score = confidence.item() * 100
    
    return label, confidence_score


def classify_with_simclr(model, probe, image_note):
    """Run SimCLR classification."""
    tensor = prepare_for_simclr(image_note)
    with torch.no_grad():
        features = model.get_features(tensor)
        outputs = probe(features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    label = CLASS_LABELS[predicted.item()]
    confidence_score = confidence.item() * 100
    
    return label, confidence_score


def classify_with_sift(classifier, image_note):
    """Run SIFT classification."""
    sift_input = prepare_for_sift(image_note)
    label, confidence, _ = classifier.predict(sift_input)
    
    if label == "unknown":
        return "Unknown", confidence * 100
    else:
        return label, confidence * 100


# ============================================
# ENSEMBLE VOTING (Majority with confidence tie-breaker, no threshold)
# ============================================

def ensemble_predict(resnet_result, simclr_result, sift_result):
    """
    Ensemble voting: majority vote (no threshold).
    All models' predictions are included.
    Tie-breaker: highest confidence.
    """
    votes = []
    vote_details = {}
    
    # Add all predictions (no threshold check)
    if resnet_result:
        votes.append({'label': resnet_result[0], 'confidence': resnet_result[1], 'model': 'ResNet-18'})
        vote_details['ResNet-18'] = resnet_result
    
    if simclr_result:
        votes.append({'label': simclr_result[0], 'confidence': simclr_result[1], 'model': 'SimCLR'})
        vote_details['SimCLR'] = simclr_result
    
    if sift_result and sift_result[0] != "Unknown":
        votes.append({'label': sift_result[0], 'confidence': sift_result[1], 'model': 'SIFT-FLANN'})
        vote_details['SIFT-FLANN'] = sift_result
    
    if not votes:
        return "Unknown", 0.0, vote_details, "No votes"
    
    # Count votes by label
    label_counts = {}
    label_confidences = {}
    
    for vote in votes:
        label = vote['label']
        if label not in label_counts:
            label_counts[label] = 0
            label_confidences[label] = []
        label_counts[label] += 1
        label_confidences[label].append(vote['confidence'])
    
    # Find label with most votes
    max_votes = max(label_counts.values())
    top_labels = [label for label, count in label_counts.items() if count == max_votes]
    
    # If clear winner
    if len(top_labels) == 1:
        winner = top_labels[0]
        avg_conf = np.mean(label_confidences[winner])
        result_type = "UNANIMOUS" if max_votes == len(votes) else "MAJORITY"
    else:
        # Tie-breaker: highest average confidence
        best_label = None
        best_avg_conf = 0
        for label in top_labels:
            avg_conf = np.mean(label_confidences[label])
            if avg_conf > best_avg_conf:
                best_avg_conf = avg_conf
                best_label = label
        winner = best_label
        avg_conf = best_avg_conf
        result_type = "TIE BREAKER"
    
    return winner, avg_conf, vote_details, result_type


# ============================================
# GUI LAYOUT
# ============================================

st.title("💰 South African Banknote Recognition")
st.markdown("Upload a banknote image to identify its denomination (R10, R20, R50, R100, R200)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_options = ["ResNet-18", "SimCLR", "SIFT-FLANN", "🔮 Ensemble (All 3 Models)"]
    selected_model = st.selectbox("Select Model", model_options)
    
    st.divider()
    
    # Model status
    st.subheader("📦 Model Status")
    
    resnet_model, resnet_acc = load_resnet_model()
    simclr_model, simclr_probe, simclr_acc = load_simclr_model()
    sift_classifier, sift_refs = load_sift_classifier()
    
    col1, col2 = st.columns(2)
    with col1:
        if resnet_model:
            st.success("✅ ResNet-18")
        else:
            st.error("❌ ResNet-18")
    
    with col2:
        if simclr_model:
            st.success("✅ SimCLR")
        else:
            st.error("❌ SimCLR")
    
    if sift_classifier:
        st.success(f"✅ SIFT-FLANN ({sift_refs} refs)")
    else:
        st.error("❌ SIFT-FLANN")

# Main area
st.subheader("📸 Upload Banknote Image")

uploaded_file = st.file_uploader(
    "Drag and drop or click to browse",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image of a South African banknote"
)

col1, col2 = st.columns([1, 1])
with col1:
    clear_button = st.button("🗑️ Clear", use_container_width=True)
with col2:
    classify_button = st.button("🔍 Classify", type="primary", use_container_width=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if classify_button:
        with st.spinner("Processing image..."):
            start_time = time.time()
            
            try:
                # Preprocess
                image_note = preprocess_image(image_array, use_clahe=False)
                
                with st.expander("View Preprocessed Note"):
                    st.image(image_note, caption="After Preprocessing & Segmentation", 
                            use_container_width=True, clamp=True)
                
                # Single model mode (no threshold)
                if selected_model == "ResNet-18":
                    if resnet_model is None:
                        st.error("ResNet model not available.")
                    else:
                        label, confidence = classify_with_resnet(resnet_model, image_note)
                        st.success(f"### 🎯 Prediction: **{label}**")
                        st.metric("Confidence", f"{confidence:.1f}%")
                
                elif selected_model == "SimCLR":
                    if simclr_model is None:
                        st.error("SimCLR model not available.")
                    else:
                        label, confidence = classify_with_simclr(simclr_model, simclr_probe, image_note)
                        st.success(f"### 🎯 Prediction: **{label}**")
                        st.metric("Confidence", f"{confidence:.1f}%")
                
                elif selected_model == "SIFT-FLANN":
                    if sift_classifier is None:
                        st.error("SIFT classifier not available.")
                    else:
                        label, confidence = classify_with_sift(sift_classifier, image_note)
                        if label == "Unknown":
                            st.warning(f"⚠️ Not a banknote or unclear image")
                            st.metric("Confidence", f"{confidence:.1f}%")
                        else:
                            st.success(f"### 🎯 Prediction: **{label}**")
                            st.metric("Confidence", f"{confidence:.1f}%")
                
                else:  # Ensemble mode
                    st.info("🔮 Running all 3 models for ensemble voting...")
                    
                    # Get predictions from all models
                    resnet_result = None
                    simclr_result = None
                    sift_result = None
                    
                    if resnet_model:
                        label, conf = classify_with_resnet(resnet_model, image_note)
                        resnet_result = (label, conf)
                        st.write(f"  ResNet-18: {label} ({conf:.1f}%)")
                    
                    if simclr_model:
                        label, conf = classify_with_simclr(simclr_model, simclr_probe, image_note)
                        simclr_result = (label, conf)
                        st.write(f"  SimCLR: {label} ({conf:.1f}%)")
                    
                    if sift_classifier:
                        label, conf = classify_with_sift(sift_classifier, image_note)
                        sift_result = (label, conf)
                        st.write(f"  SIFT-FLANN: {label} ({conf:.1f}%)")
                    
                    # Ensemble voting (no threshold)
                    final_label, final_conf, vote_details, result_type = ensemble_predict(
                        resnet_result, simclr_result, sift_result
                    )
                    
                    st.divider()
                    
                    if final_label == "Unknown":
                        st.warning(f"⚠️ No valid predictions")
                        st.error("Not a banknote or unclear image")
                    else:
                        st.success(f"### 🎯 Ensemble Prediction: **{final_label}**")
                        st.metric("Confidence", f"{final_conf:.1f}%")
                        st.caption(f"Verdict: {result_type}")
                        
                        # Show individual results
                        st.subheader("🗳️ Individual Model Votes")
                        col_r, col_s, col_sift = st.columns(3)
                        
                        with col_r:
                            if resnet_result:
                                st.info(f"**ResNet-18**\n\n{resnet_result[0]}\n({resnet_result[1]:.1f}%)")
                            else:
                                st.info(f"**ResNet-18**\n\nNot loaded")
                        
                        with col_s:
                            if simclr_result:
                                st.info(f"**SimCLR**\n\n{simclr_result[0]}\n({simclr_result[1]:.1f}%)")
                            else:
                                st.info(f"**SimCLR**\n\nNot loaded")
                        
                        with col_sift:
                            if sift_result:
                                st.info(f"**SIFT-FLANN**\n\n{sift_result[0]}\n({sift_result[1]:.1f}%)")
                            else:
                                st.info(f"**SIFT-FLANN**\n\nNot loaded")
                
                elapsed_time = time.time() - start_time
                st.caption(f"⏱️ Inference time: {elapsed_time:.2f} seconds")
                
            except Exception as e:
                st.error(f"Error during classification: {e}")

if clear_button:
    st.rerun()

st.divider()
st.caption("South African Banknote Recognition System | Models: ResNet-18, SimCLR, SIFT-FLANN")