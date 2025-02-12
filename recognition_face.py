#!/usr/bin/python3

import cv2
import joblib
import numpy as np
from picamera2 import Picamera2

# Load the trained SVM model
model = joblib.load("face_recognition_model.pkl")

# Load OpenCV's face detection model
face_detector = cv2.CascadeClassifier("/home/nano/Face_Recognition/haarcascade_frontalface_default.xml")

# Initialize PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Confidence threshold (ONLY print names if confidence > 0.6)
CONFIDENCE_THRESHOLD = 0.6

print("🎥 Live Face Recognition Started... Press 'q' to exit.")

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    detected_names = []  # List to store detected names

    for (x, y, w, h) in faces:
        # Extract face ROI and resize to match training images
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        face_flat = face_img.flatten().reshape(1, -1)

        # Predict identity
        predicted_label = model.predict(face_flat)[0]
        confidence_scores = model.predict_proba(face_flat)[0]  # Get probabilities
        confidence_score = confidence_scores.max()  # Highest probability score

        # Show name only if confidence is above 0.6
        if confidence_score >= CONFIDENCE_THRESHOLD:
            detected_names.append(predicted_label)
            # Draw rectangle and name on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Mark unknown faces but don't print them
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Print detected names **only if confidence > 0.6**
    if detected_names:
        print("Detected Faces:", ", ".join(detected_names))

    # Show the frame in real-time
    cv2.imshow("Live Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.close()
print("🚪 Exiting live recognition.")
