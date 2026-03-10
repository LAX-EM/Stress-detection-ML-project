import os
import cv2
import numpy as np
import joblib
import base64
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load Models
emotion_model = load_model('emotion_model.h5', compile=False)
stress_model = joblib.load('stress_model.pkl')
scaler = joblib.load('scaler.pkl')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
mood_map = {'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'surprise': 4}

def get_emotion_from_base64(base64_str):
    # Decode base64 image
    encoded_data = base64_str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Resize and Format for .h5 model
    img = cv2.resize(img, (64, 64))
    img = img.astype('float') / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) # Shape (1, 64, 64, 1)
    
    preds = emotion_model.predict(img, verbose=0)[0]
    return emotion_labels[np.argmax(preds)].lower()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get Image Data and Manual Features
    image_data = request.form['image_data']
    sleep = float(request.form['sleep'])
    sent = int(request.form['sent'])
    received = int(request.form['received'])
    c_in = int(request.form['c_in'])
    c_out = int(request.form['c_out'])

    # 2. Identify Emotion
    emotion = get_emotion_from_base64(image_data)
    
    # 3. Map Emotion to Int
    mood_int = mood_map.get(emotion, 1)
    if emotion in ['disgust', 'fear']: mood_int = 3

    # 4. Predict Stress (Target Variable)
    features = [[mood_int, sleep, sent, received, c_in, c_out]]
    scaled_features = scaler.transform(features)
    prediction = stress_model.predict(scaled_features)[0]
    
    stress_levels = ["Low", "Medium", "High"]
    target_variable = stress_levels[prediction]

    return render_template('index.html', 
                           emotion=emotion.capitalize(), 
                           stress_level=target_variable)

if __name__ == '__main__':
    app.run(debug=True)