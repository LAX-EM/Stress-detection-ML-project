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
