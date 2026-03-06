import numpy as np
import os
from skimage.feature import hog
from skimage import transform
from skimage import io as skimage_io

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

def _read_and_resize(image_path, size=(128, 128)):
    # Pehle check karein ki file exist karti hai ya nahi
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image nahi mili is path par: {image_path}")

    # --- LINE 18 FIX: OpenCV Logic ---
    if _HAS_CV2 and cv2 is not None:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return cv2.resize(img, size)

    # --- LINE 18 FALLBACK: Agar OpenCV fail ho jaye toh Skimage chalega ---
    try:
        img = skimage_io.imread(image_path, as_gray=True)
        # resize float [0,1] return karta hai, use uint8 [0,255] mein convert karein
        img_resized = transform.resize(img, size, anti_aliasing=True)
        return (img_resized * 255).astype(np.uint8)
    except Exception as e:
        raise FileNotFoundError(f"Dono libraries image read nahi kar payi: {e}")

def extract_features(image_path):
    # --- LINE 40: Image data yahan preprocess hota hai ---
    img = _read_and_resize(image_path, (128, 128))

    # HOG features nikalna
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features