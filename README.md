# RaspberryPi-FaceRecognition
Face Recognition on Raspberry Pi, A real-time face recognition system using: Raspberry Pi, OpenCV, and SVM classification. 

This project implements **real-time face recognition** using a **Raspberry Pi Camera Module 2** and **OpenCV**. It captures faces, trains an SVM model, and recognizes known faces in a live video feed.  

## **Features**  
✅ Face detection using OpenCV’s Haar cascades  
✅ Face recognition using an SVM model with HOG features  
✅ Live video feed with detected names displayed on the screen  
✅ Unrecognized faces labeled as **"Unknown"**  
✅ Works on **Raspberry Pi with Picamera2**  

## **Installation**  

### **1. Install Dependencies**  
Run the following commands to install the required libraries:  
```bash
sudo apt update && sudo apt upgrade -y
pip3 install opencv-contrib-python numpy scikit-learn picamera2
```

### **2. Enable Camera on Raspberry Pi**  
Ensure the Raspberry Pi camera is enabled:  
```bash
sudo raspi-config
```
- **Go to "Interface Options" → "Camera" → Enable**  
- **Disable the legacy camera support**  

Reboot the Raspberry Pi:  
```bash
sudo reboot
```

## **Usage**  

### **1. Capture Face Images**  
Run the script to capture images for training:  
```bash
python3 capture_faces.py
```
- Enter the person's name when prompted.  
- The script will capture 10 images per person and store them in `captured_faces/`.  

### **2. Train the Face Recognition Model**  
After capturing images, train the model:  
```bash
python3 train_faces.py
```
- This script extracts HOG features, trains an **SVM classifier**, and saves the model (`face_svm_model.pkl`).  

### **3. Live Face Recognition**  
Run the script to recognize faces in real-time:  
```bash
python3 recognition_face.py
```
- Detected faces will be **displayed on the video feed with names**.  
- If a face is not recognized, it will be labeled as **"Unknown"**.  

## **Demo**  
![Demo](https://via.placeholder.com/600x300?text=Face+Recognition+Demo)  

## **License**  
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details. 
