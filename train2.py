#!/usr/bin/python3

import cv2
import numpy as np
import os
import joblib
from sklearn.svm import SVC

# Load training images
def load_images(data_dir):
    X, y, label_map = [], [], {}
    label_id = 0

    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        
        if person_name not in label_map:
            label_map[person_name] = label_id
            label_id += 1

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (100, 100)).flatten()
            X.append(img)
            y.append(label_map[person_name])

    return np.array(X), np.array(y), label_map

# Load images for training
data_dir = "captured_faces"
X_train, y_train, label_mapping = load_images(data_dir)

# Train SVM model
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# Save model and labels
joblib.dump(svm_model, "face_svm_model.pkl")
joblib.dump(label_mapping, "label_mapping.pkl")

print("✅ Training completed! Model saved as 'face_svm_model.pkl'")
