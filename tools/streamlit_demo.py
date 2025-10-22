#!/usr/bin/env python
# ------------------------------------------------------------------------------
# FiDIP Streamlit Demo
# Copyright (c) 2025 Augmented Cognition Lab, Northeastern University
# Licensed under The Apache-2.0 License
# ------------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
import argparse
import subprocess

# Add lib to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# COCO keypoint names
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO skeleton definition (pairs of keypoint indices that form a line)
SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # Face
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Upper body
    [5, 11], [6, 12], [11, 12],  # Torso
    [11, 13], [13, 15], [12, 14], [14, 16]  # Lower body
]

# Colors in RGB format for Streamlit
KEYPOINT_COLOR = (255, 0, 0)  # Red
SKELETON_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 0, 0)      # Red


def run_demo_script(image_path, output_path, config_path, model_path, conf_threshold=0.3, output_size="512,512"):
    """Run the demo.py script as a subprocess to avoid PyTorch path issues"""
    cmd = [
        "python", "tools/demo.py",
        "--cfg", config_path,
        "--image", image_path,
        "--model", model_path,
        "--output", output_path,
        "--conf-threshold", str(conf_threshold),
        "--output-size", output_size
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return False, stderr
        
        return True, stdout
    except Exception as e:
        return False, str(e)


def save_uploaded_file(uploaded_file):
    """Save the uploaded file to disk"""
    # Create a temporary directory if it doesn't exist
    temp_dir = os.path.join(os.getcwd(), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def main():
    st.set_page_config(
        page_title="FiDIP Infant Pose Estimation",
        page_icon="👶",
        layout="wide"
    )
    
    st.title("FiDIP Infant Pose Estimation")
    st.write("Upload an image to estimate infant pose")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Model configuration
    config_path = st.sidebar.text_input(
        "Config Path", 
        value="experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml"
    )
    
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="models/hrnet_fidip.pth"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.05
    )
    
    output_size = st.sidebar.text_input(
        "Output Size", 
        value="512,512"
    )
    
    # Check if paths exist
    if not os.path.exists(config_path):
        st.sidebar.error(f"Config file not found: {config_path}")
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file not found: {model_path}")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Process image if uploaded
    if uploaded_file is not None and os.path.exists(config_path) and os.path.exists(model_path):
        # Save the uploaded file
        input_path = save_uploaded_file(uploaded_file)
        
        # Create output path
        output_dir = os.path.join(os.getcwd(), "temp_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{os.path.basename(input_path)}")
        conf_output_path = os.path.join(output_dir, f"result_{os.path.basename(input_path).split('.')[0]}_conf.jpg")
        
        # Display original image
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image
        with st.spinner("Processing image..."):
            success, log = run_demo_script(
                input_path, 
                output_path, 
                config_path, 
                model_path,
                confidence_threshold,
                output_size
            )
            
            if success and os.path.exists(output_path):
                # Display results
                st.subheader("Pose Estimation Results")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    result_image = Image.open(output_path)
                    st.image(result_image, caption="Pose Estimation", use_column_width=True)
                
                with col2:
                    if os.path.exists(conf_output_path):
                        conf_image = Image.open(conf_output_path)
                        st.image(conf_image, caption="Keypoint Confidence", use_column_width=True)
                    else:
                        st.write("Confidence visualization not available")
                
                # Display processing log
                with st.expander("Processing Log"):
                    st.text(log)
            else:
                st.error(f"Error processing image: {log}")
    
    # Instructions
    if uploaded_file is None:
        st.info("Please upload an image to start pose estimation.")


if __name__ == "__main__":
    main()
