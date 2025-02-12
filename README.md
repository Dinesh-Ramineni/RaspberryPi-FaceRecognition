# RaspberryPi-FaceRecognition

A real-time face recognition system using Raspberry Pi and PiCamera2. This project captures, trains, and recognizes faces with SVM classification and OpenCV.

## Features
- Live face detection and recognition
- Captures images and trains an SVM model
- Recognizes known faces and labels unrecognized faces as "Unknown"
- Runs efficiently on Raspberry Pi using PiCamera2

## Hardware Requirements
- Raspberry Pi (Tested on Raspberry Pi 4)
- Raspberry Pi Camera Module 2
- MicroSD Card (32GB recommended)
- Power Adapter

## Software Requirements
- Raspberry Pi OS (Bullseye or newer)
- Python 3.x
- OpenCV (with `opencv-contrib-python` for face recognition)
- Scikit-learn
- PiCamera2 Library

## Installation
Clone the repository and install dependencies:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip libopencv-dev
pip3 install -r requirements.txt
```

### Enable Camera
Make sure the legacy camera interface is **disabled** and the new driver is enabled:

```bash
sudo raspi-config
```
Go to **Interfacing Options → Camera → Disable** (for legacy)

## Usage
### 1. Capture Faces
Run the following command to capture face images:
```bash
python3 capture_faces.py
```
Enter the person's name when prompted. It will save 10 images in `captured_faces/{person_name}`.

### 2. Train the Model
```bash
python3 train_faces.py
```
This will generate `face_svm_model.pkl` in the models directory.

### 3. Recognize Faces in Real-Time
```bash
python3 recognition_face.py
```
The camera window will show recognized faces with their names. Press `q` to exit.

## Troubleshooting
- **Camera Not Found Error:** Run `sudo raspi-config` and enable the new camera driver.
- **Module Not Found:** Ensure dependencies are installed (`pip3 install -r requirements.txt`).
- **Low Accuracy:** Try capturing more images per person before training.

## License
This project is licensed under the [MIT License](LICENSE).

## Contributing
Feel free to fork and contribute! Open an issue if you face any problems.

---
**Author:** Dinesh Ramineni

