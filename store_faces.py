#!/usr/bin/python3

import cv2
import os  
import time
from picamera2 import Picamera2

# Load OpenCV's face detection model
face_detector = cv2.CascadeClassifier("/home/nano/Face_Recognition/haarcascade_frontalface_default.xml")

# Initialize PiCamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Ask for the person's name
person_name = input("Enter person's name: ")

# Create a directory to store detected faces
output_directory = os.path.join("captured_faces", person_name)
os.makedirs(output_directory, exist_ok=True)

count = 0  # Counter for saved images

while count < 30:  # Capture 10 images per person
    frame = picam2.capture_array()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save only the detected face portion
        face_crop = frame[y:y+h, x:x+w]

        # Generate a unique filename
        filename = os.path.join(output_directory, f"img{count:02d}.jpg")
        cv2.imwrite(filename, face_crop)  
        print(f"Saved: {filename}")

        count += 1
        if count >= 10:
            break  # Stop after capturing 10 images

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(500) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cv2.destroyAllWindows()
picam2.close()
print(f"✅ Face capturing completed for {person_name}")
