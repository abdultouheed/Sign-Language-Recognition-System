# 🤟 Sign Language Recognition System

A real-time **Sign Language Recognition System** developed using **Python, OpenCV, Computer Vision, and Deep Learning** to recognize English sign language gestures through a webcam.

The system captures hand gestures from a live camera feed, processes the input, recognizes the corresponding gesture, and provides the prediction as text. The project also includes a graphical interface for interacting with the recognition system.

---

## 📌 Project Overview

Communication can be challenging for people who rely on sign language when interacting with individuals who do not understand it.

This project aims to provide an AI-based solution that can recognize sign language gestures from webcam input and convert the recognized gestures into understandable text.

The system follows a real-time computer vision pipeline:

```text
Webcam
   ↓
Capture Hand Gesture
   ↓
Image / Gesture Processing
   ↓
Feature Extraction
   ↓
Deep Learning / Classification
   ↓
Gesture Recognition
   ↓
Text Output
```

---

## ✨ Features

* 🎥 Real-time webcam-based gesture recognition
* 🤟 English sign language gesture recognition
* 🧠 Deep Learning-based classification
* 👁️ Computer Vision processing using OpenCV
* 📝 Converts recognized gestures into text
* 🖥️ Graphical user interface
* ⚡ Real-time prediction from camera input
* ✋ Recognition under different hand positions

---

## 🏗️ System Workflow

```text
                 ┌─────────────────┐
                 │     Webcam      │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Capture Frames  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Gesture / Hand  │
                 │   Processing    │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Feature / Image │
                 │   Processing    │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Classification  │
                 │     Model       │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Gesture / Sign  │
                 │   Prediction    │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  Text Output    │
                 └─────────────────┘
```

---

## 📂 Project Structure

```text
Sign-Language-Recognition-System/
│
├── camera1.py
├── gest.py
├── gui.py
├── slr.py
├── testslr.py
└── README.md
```

### File Description

| File         | Description                                                          |
| ------------ | -------------------------------------------------------------------- |
| `camera1.py` | Handles camera/webcam-related functionality for capturing live input |
| `gest.py`    | Contains gesture-related processing and recognition functionality    |
| `gui.py`     | Provides the graphical user interface for the application            |
| `slr.py`     | Main sign language recognition functionality                         |
| `testslr.py` | Used for testing the sign language recognition system                |
| `README.md`  | Project documentation                                                |

---

## 🧠 Recognition Process

The system processes the webcam input through multiple stages.

### 1. Camera Input

The webcam captures live frames containing the user's hand gesture.

### 2. Image Processing

The captured frames are processed using computer vision techniques to prepare the gesture for recognition.

### 3. Gesture Recognition

The processed hand gesture is passed through the recognition pipeline to identify the corresponding sign.

### 4. Classification

The system determines which trained gesture/class the input corresponds to.

### 5. Text Output

The recognized sign is displayed as text, allowing the gesture to be interpreted by the user.

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **Deep Learning**
* **Computer Vision**
* **Machine Learning**
* **GUI Development**

---

## ▶️ Running the Project

Clone the repository:

```bash
git clone https://github.com/abdultouheed/Sign-Language-Recognition-System.git
```

Navigate to the project directory:

```bash
cd Sign-Language-Recognition-System
```

Run the required Python file:

```bash
python slr.py
```

Depending on the implementation, the GUI or camera functionality can also be started through the corresponding Python files.

---

## 🎯 Applications

The system can be used for:

* Accessibility and assistive communication
* Sign language learning
* Human-computer interaction
* Educational applications
* Gesture-controlled systems
* Communication assistance
* Computer vision research

---

## 📈 Project Highlights

* Developed a **real-time computer vision system** for sign language recognition.
* Implemented **webcam-based gesture detection and classification**.
* Applied **Deep Learning and OpenCV** for gesture recognition.
* Designed a **GUI-based interface** for interacting with the application.
* Converted recognized gestures into **readable text**.
* Built separate modules for camera handling, gesture processing, recognition, GUI, and testing.

---

## 🚀 Future Improvements

The system can be further improved by:

* Supporting a larger sign language vocabulary
* Recognizing complete words and sentences
* Implementing continuous sign language recognition
* Improving recognition under different lighting conditions
* Supporting multiple hand gestures
* Adding text-to-speech functionality
* Improving model accuracy with a larger dataset
* Deploying the system as a web or mobile application

---

## 👨‍💻 Author

**Abdul Touheed**

Computer Science Engineer | Machine Learning Enthusiast | Python Developer

---
