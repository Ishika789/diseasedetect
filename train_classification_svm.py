import os
from sklearn.svm import SVC
import joblib
from features.feature_extraction import extract_features

def load_data(path):
    X, y = [], []
    for label in os.listdir(path):
        folder = os.path.join(path, label)
        if not os.path.isdir(folder): continue
        for img in os.listdir(folder):
            X.append(extract_features(os.path.join(folder, img)))
            y.append(label)
    return X, y

# Data load karo, train karo aur save karo
X, y = load_data("dataset")
svm = SVC()
svm.fit(X, y)
joblib.dump(svm, "models/classification_svm.pkl")