import os
import sys
import numpy as np # type: ignore
import joblib # type: ignore
from pathlib import Path

# --- 1. IMPORTS & SETUP ---
from skimage.feature import hog # type: ignore
from skimage import transform # type: ignore
from skimage import io as skimage_io # type: ignore

try:
    import cv2 # type: ignore
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

# --- 2. IMAGE PROCESSING (Fixes for Line 18 & 40) ---

def _read_and_resize(image_path, size=(128, 128)):
    # Physical check: Kya file sach mein wahan hai?
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Python ko is path par image nahi mili: {image_path}")

    img = None

    # Step A: Pehle OpenCV try karein (Line 18 Logic)
    if _HAS_CV2 and cv2 is not None:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # Step B: Fallback to Skimage (Agar OpenCV fail ho jaye)
    try:
        img = skimage_io.imread(image_path, as_gray=True)
        img_resized = transform.resize(img, size, anti_aliasing=True)
        # HOG ke liye float [0,1] ko uint8 [0,255] mein convert karna zaroori hai
        return (img_resized * 255).astype(np.uint8)
    except Exception as e:
        raise FileNotFoundError(f"Dono libraries (CV2/Skimage) image read nahi kar payi: {e}")

def extract_features(image_path):
    # Line 40: Processing function ko call karna
    img = _read_and_resize(image_path, (128, 128))

    # HOG feature extraction
    # 
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

# --- 3. MAIN LOGIC (Fix for Line 25) ---

def run_project(file_path):
    # Models load karein (Ensure 'models' folder is in 'diseasedetect' folder)
    try:
        detection_model = joblib.load("models/detection_svm.pkl")
        classification_model = joblib.load("models/classification_svm.pkl")
    except Exception as e:
        return f"Error: Model file nahi mili. Check 'models' folder. ({e})"

    # Feature extraction aur Reshape (Line 25 Logic)
    try:
        features = extract_features(file_path).reshape(1, -1)
    except Exception as e:
        return f"Error during feature extraction: {e}"

    # Prediction
    detection = detection_model.predict(features)[0]

    if detection == "normal":
        return "Result: Lung is Normal"

    disease = classification_model.predict(features)[0]
    return f"Result: Disease Detected -> {disease}"

# --- 4. EXECUTION ---

if __name__ == "__main__":
    # YAHAN AAPKA PATH ADD KIYA HAI:
    # Windows paths ke liye hamesha r"" (raw string) use karein
    MY_IMAGE_PATH = r"C:\Users\Ishika\Desktop\diseasedetect\dataset\hetero\sample.jpeg"

    print(f"Testing Image: {MY_IMAGE_PATH}")
    
    output = run_project(MY_IMAGE_PATH)
    
    print("-" * 40)
    print(output)
    print("-" * 40)