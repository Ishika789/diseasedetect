import os
import sys
import subprocess

try:
    import numpy as np
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        import numpy as np
    except Exception as e:
        raise ImportError("Failed to import or install numpy: " + str(e))

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from features.feature_extraction import extract_features

def load_data(path):
    X, y = [], []
    for label in os.listdir(path):
        folder = os.path.join(path, label)
        for img in os.listdir(folder):
            features = extract_features(os.path.join(folder, img))
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data("dataset")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train, y_train)

print("Detection Accuracy:", accuracy_score(y_test, svm.predict(X_test)))

joblib.dump(svm, "models/detection_svm.pkl")
