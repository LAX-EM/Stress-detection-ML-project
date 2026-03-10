import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. LOAD DATA
print("📂 Loading dataset...")
df = pd.read_csv("mood_dataset_9673.csv")

# 2. PREPROCESSING
# Your specified mood mapping
mood_map = {'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'surprised': 4}

# Ensure column names match your CSV exactly (lowercase/strip whitespace)
df['face_mood'] = df['face_mood'].str.lower().str.strip().map(mood_map)

# Map stress level to integers
stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['stress_level'] = df['stress_level'].map(stress_map)

# Handle any NaN values (just in case)
df = df.dropna()

# 3. DEFINE FEATURES AND TARGET
X = df[['face_mood', 'sleep_hours', 'messages_sent', 'messages_received', 'calls_incoming', 'calls_outgoing']]
y = df['stress_level']

# 4. DATA SCALING
# Scaler is crucial because 'messages' (0-50) are much larger than 'sleep' (0-9)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. TRAIN MODEL
print("🧠 Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. EVALUATE
y_pred = model.predict(X_test)
print("\n--- Model Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# 8. SAVE EVERYTHING
# You MUST save the scaler along with the model for your webcam script to work
joblib.dump(model, 'stress_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ Training Complete! 'stress_model.pkl' and 'scaler.pkl' are ready.")