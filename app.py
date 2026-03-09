import streamlit as st  # type: ignore
import os
from PIL import Image # type: ignore
import numpy as np # type: ignore
import requests # type: ignore
import folium # type: ignore
import google.generativeai as genai  # type: ignore
from flask import Flask, request, jsonify, render_template # type: ignore


# Import your existing logic from your script
# Assuming your script is named 'engine.py'
from engine import run_project 

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Lung Disease Detector", layout="centered")

st.title("🫁 Lung Disease Detection System")
st.markdown("Upload a chest X-ray image to detect abnormalities and identify diseases.")

# --- SIDEBAR / INFO ---
with st.sidebar:
    st.header("About")
    st.info("This system uses HOG (Histogram of Oriented Gradients) features and SVM classifiers to analyze lung images.")
    st.warning("Note: This is an AI prototype and not a replacement for professional medical advice.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Save the uploaded file temporarily so the backend can read it via path
    temp_path = os.path.join("temp_upload.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Display the image
    cols = st.columns(2)
    with cols[0]:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # 3. Prediction Button
    with cols[1]:
        st.subheader("Analysis")
        if st.button("Run Diagnostic Scan"):
            with st.spinner('Extracting HOG features and classifying...'):
                try:
                    # Calling your run_project function
                    result = run_project(temp_path)
                    
                    # Displaying results with some styling
                    if "Normal" in result:
                        st.success(result)
                    else:
                        st.error(result)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

else:
    st.info("Please upload an X-ray image (JPEG/PNG) to begin.")

   