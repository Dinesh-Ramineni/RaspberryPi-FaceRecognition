#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-opencv libopencv-dev
pip3 install --upgrade pip
pip3 install numpy scikit-learn joblib picamera2

# Ensure the camera module is enabled
echo "Enabling camera module..."
sudo raspi-config nonint do_camera 0

# Clone the repository (if not already cloned)
if [ ! -d "RaspberryPi-FaceRecognition" ]; then
    git clone https://github.com/Dinesh-Ramineni/RaspberryPi-FaceRecognition.git
    cd RaspberryPi-FaceRecognition
else
    cd RaspberryPi-FaceRecognition
    git pull origin main
fi

# Set execute permissions for Python scripts
chmod +x *.py

# Notify user
echo "✅ Setup completed successfully! You can now run face recognition."
echo "Run: python3 capture_faces.py to start capturing images."
echo "Then run: python3 train_model.py to train."
echo "Finally, use: python3 recognition_face.py to recognize faces in real-time."
