# 🤟 Sign Language Recognition System

A real-time **Sign Language Recognition System** that uses **Deep Learning, Computer Vision, and OpenCV** to recognize English sign language gestures through a webcam and convert them into **text and speech**.

The system captures hand gestures in real time, classifies the gesture using a trained deep learning model, and provides the predicted sign as text along with optional speech output.

---

## 📌 Features

- 🎥 Real-time gesture recognition using a webcam
- 🤟 Recognition of English sign language gestures
- 🧠 Deep Learning-based gesture classification
- 👁️ Computer Vision processing using OpenCV
- 📝 Converts recognized gestures into text
- 🔊 Converts predicted text into speech
- ✋ Works with different hand positions and orientations
- ⚡ Provides live predictions through the webcam

---

## 🏗️ System Workflow

```text
Webcam
   ↓
Capture Hand Gesture
   ↓
Image Preprocessing
   ↓
Hand / Gesture Detection
   ↓
Deep Learning Model
   ↓
Gesture Classification
   ↓
Predicted Sign
   ↓
Text Output
   ↓
Speech Output

Sign-Language-Recognition/
│
├── dataset/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── ...
│
├── model/
│   └── trained_model.h5
│
├── training/
│   └── train_model.py
│
├── src/
│   ├── prediction.py
│   ├── preprocessing.py
│   └── text_to_speech.py
│
├── main.py
├── requirements.txt
└── README.md

⚙️ Installation
1. Clone the Repository
git clone https://github.com/your-username/sign-language-recognition.git

2. Navigate to the Project
cd sign-language-recognition

3. Create a Virtual Environment
python -m venv venv
Activate the environment on Windows:
venv\Scripts\activate
On Linux/macOS:
source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt
▶️ Running the Project
Start the application using:
python main.py

📊 Recognition Pipeline

          ┌──────────────┐
          │    Webcam    │
          └──────┬───────┘
                 ↓
        ┌──────────────────┐
        │ Image Processing │
        └────────┬─────────┘
                 ↓
        ┌──────────────────┐
        │ Gesture Detection│
        └────────┬─────────┘
                 ↓
        ┌──────────────────┐
        │ Deep Learning    │
        │ Classification   │
        └────────┬─────────┘
                 ↓
        ┌──────────────────┐
        │ Gesture / Letter │
        └────────┬─────────┘
                 ↓
          ┌──────┴──────┐
          ↓             ↓
     ┌─────────┐   ┌───────────┐
     │  Text   │   │   Speech  │
     └─────────┘   └───────────┘
