# 🧠 AI Stress Predictor: Multi-Modal Detection
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent web application that identifies user stress levels by combining **Real-time Facial Emotion Recognition** (Computer Vision) with **User-input Lifestyle Data** (Sleep, Communication, and Activity).



---

## 📌 Features
* **Live Camera Capture:** Snap a photo directly in the browser to detect your current emotional state.
* **Multi-Model Fusion:** Integrates a Convolutional Neural Network (`.h5`) for vision and a Random Forest Classifier (`.pkl`) for tabular data.
* **Interactive UI:** A clean, responsive dashboard built with Jinja2 templates and custom CSS.
* **Instant Classification:** Provides a target stress variable (Low, Medium, High) based on synchronized inputs.

---

## 🛠️ Tech Stack
* **Backend:** Flask (Python)
* **Computer Vision:** OpenCV, TensorFlow/Keras
* **Machine Learning:** Scikit-Learn, Joblib, Pandas
* **Frontend:** HTML5, CSS3, JavaScript (Webcam API)

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Create and activate a virtual environment to keep dependencies isolated:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

2. Installation
Install the required libraries using the requirements file:

Bash
pip install -r requirements.txt
3. Required Model Files
Ensure the following pre-trained files are located in the project root:

emotion_model.h5 - The CNN model for emotion detection.

stress_model.pkl - The Random Forest model for stress prediction.

scaler.pkl - The Standard Scaler used during training.

4. Running the App
Start the Flask development server:

Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000.
