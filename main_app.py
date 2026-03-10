import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 1. SETUP LABELS AND MAPPING
# Your specified labels from the emotion model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Your specified mapping for the stress model
# Note: Ensure these strings match the lowercase labels below
mood_map = {'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'surprise': 4}

# 2. LOAD MODELS & ASSETS
print("⌛ Loading AI models...")
# 'compile=False' fixes the 'lr' argument error
emotion_model = load_model('emotion_model.h5', compile=False)
stress_model = joblib.load('stress_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load face detector (OpenCV default)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. MOCKED LIFESTYLE DATA
# In a real app, these values would update from a database or UI inputs
user_sleep = 5.5
msg_sent = 30
msg_received = 28
calls_in = 5
calls_out = 4

# 4. WEBCAM LOOP
cap = cv2.VideoCapture(0)

print("✅ System Ready. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
        
        # --- PREPROCESS FACE FOR EMOTION MODEL ---
        roi_gray = gray[y:y+h, x:x+w]
        
        # Fix: Resize to 64x64 to match your model's expected input
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Normalize and prepare array
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi) # Makes it (64, 64, 1)
        roi = np.expand_dims(roi, axis=0) # Makes it (1, 64, 64, 1)

        # STEP A: Predict Emotion
        prediction = emotion_model.predict(roi, verbose=0)[0]
        detected_emotion = emotion_labels[np.argmax(prediction)].lower()
        
        # STEP B: Map emotion to Stress-Model Integer
        # Handle "Disgust" and "Fear" by mapping them to high-arousal negative categories
        if detected_emotion in mood_map:
            mood_int = mood_map[detected_emotion]
        elif detected_emotion in ['disgust', 'fear']:
            mood_int = mood_map['angry']
        else:
            mood_int = mood_map['neutral']

        # STEP C: Predict Stress
        # Input order: [mood_int, sleep, msg_sent, msg_received, calls_in, calls_out]
        feature_list = [mood_int, user_sleep, msg_sent, msg_received, calls_in, calls_out]
        
        # Apply the Scaler (Must be done before prediction)
        scaled_features = scaler.transform([feature_list])
        stress_idx = stress_model.predict(scaled_features)[0]
        
        # Map result index back to text
        stress_levels = ["LOW", "MEDIUM", "HIGH"]
        final_stress = stress_levels[stress_idx]

        # STEP D: UI Display
        # Color coding: Green for Low, Orange for Medium, Red for High
        color = (0, 255, 0) if final_stress == "LOW" else (0, 165, 255) if final_stress == "MEDIUM" else (0, 0, 255)
        
        cv2.putText(frame, f"Emotion: {detected_emotion.capitalize()}", (x, y-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"STRESS: {final_stress}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    cv2.imshow('Stress Predictor AI', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()