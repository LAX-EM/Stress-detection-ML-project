# 🧠 AI Multi-Modal Stress Predictor

An intelligent web application that identifies **user stress levels** by combining **Real-time Facial Emotion Recognition (Computer Vision)** with **user lifestyle data** such as sleep patterns, communication activity, and daily behavior.

The system integrates **multiple AI models** to analyze both **visual and behavioral signals** and produce an overall **stress classification (Low, Medium, High)**.

---

# 📌 Features

### 📷 Live Camera Capture

Capture a photo directly from the browser using the **Webcam API** to analyze facial emotions in real time.

### 🤖 Multi-Model Fusion

The system integrates two machine learning models:

* **CNN Model (.h5)** → Detects facial emotions using computer vision.
* **Random Forest Model (.pkl)** → Predicts stress using lifestyle data.

### ⚡ Instant Stress Classification

The system combines outputs from both models to classify stress levels into:

* **Low Stress**
* **Medium Stress**
* **High Stress**

### 🖥 Interactive Web Interface

A responsive dashboard built using **Jinja2 templates, HTML, CSS, and JavaScript**.

---

# 🛠 Tech Stack

## Backend

* Python
* Flask

## Computer Vision

* OpenCV
* TensorFlow / Keras

## Machine Learning

* Scikit-learn
* Joblib
* Pandas
* NumPy

## Frontend

* HTML5
* CSS3
* JavaScript
* Webcam API

---

# 🚀 Environment Setup & Installation

Follow these steps to run the project locally.

---

## 1️⃣ Clone the Repository

```Type in your terminal
git clone https://github.com/LAX-EM/Stress-detection-ML-project.git
cd Stress_ML_Project
```

---

## 2️⃣ Create a Virtual Environment

Using a virtual environment prevents dependency conflicts.

### Windows

```Type in your terminal
python -m venv venv
venv\Scripts\activate
```

### Mac / Linux

```Type in your terminal
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Required Packages

Install all required dependencies:

```Type in your terminal
pip install flask opencv-python tensorflow scikit-learn joblib pandas numpy Pillow
```

---

## 4️⃣ Verify Model Files

Ensure the following trained models exist in the **project root directory**:

```
emotion_model.h5     # Facial emotion recognition model
stress_model.pkl     # Stress prediction model
scaler.pkl           # Feature scaler
```

---

# 🖥 Running the Application

## Step 1: Start the Server

Run the Flask application:

```Type in your terminal
python app.py
```

---

## Step 2: Open the Web Application

Open your browser and go to:

```
http://127.0.0.1:5000
```

Then:

1. Allow **webcam access** in the browser.
2. Enter **lifestyle details** (sleep hours, messages, activity, etc.).
3. Click **"Capture & Identify Stress"**.

The system will analyze both **facial emotion** and **lifestyle data** to determine the **stress level**.

---

# 📊 Output

The system classifies stress into three levels:

| Stress Level | Description              |
| ------------ | ------------------------ |
| Low          | Normal emotional state   |
| Medium       | Moderate stress detected |
| High         | High stress detected     |

---

# 📁 Project Structure

```
Stress_ML_Project
│
├── app.py
├── emotion_model.h5
├── stress_model.pkl
├── scaler.pkl
│
├── templates
│   └── index.html
│
├── static
│   ├── css
│   └── js
│
└── README.md
```

---

# 👨‍💻 Author

Developed as an **AI + Computer Vision project** demonstrating **multi-modal stress detection** using **machine learning models and real-time facial emotion recognition**.
