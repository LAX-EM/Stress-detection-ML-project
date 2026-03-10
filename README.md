🧠 AI-Driven Stress DetectorAn end-to-end Machine Learning project that predicts human stress levels by combining Real-time Facial Emotion Recognition (Computer Vision) and Simulated Lifestyle Data (Mobile Usage/Sleep).🚀 How it WorksEmotion AI: Uses a CNN (.h5) model to detect facial expressions via webcam.Data AI: Uses a Random Forest (.pkl) model to analyze lifestyle factors (Sleep, Messages, Calls).Final Output: A Flask web interface that displays a "Target Variable" (Low, Medium, or High Stress).🛠️ Environment Setup (Step-by-Step)1. Clone the RepositoryBashgit clone https://github.com/your-username/Stress_ML_Project.git
cd Stress_ML_Project
2. Create a Virtual EnvironmentThis keeps your project libraries isolated and prevents version conflicts.Bash# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install DependenciesInstall all required AI and Web libraries using the requirements file:Bashpip install -r requirements.txt
🏗️ Project ArchitectureFileRoleapp.pyThe Flask backend server that handles AI logic.templates/index.htmlThe Jinja2 frontend with webcam integration.emotion_model.h5Deep Learning model for face emotion detection.stress_model.pklScikit-Learn model for lifestyle stress prediction.scaler.pklThe feature scaling object for data normalization.🖥️ Running the ApplicationStep 1: Prepare the ModelsEnsure your .h5 and .pkl files are in the root directory.Step 2: Start the Flask ServerBashpython app.py
Step 3: Access the UIOpen your browser and go to: http://127.0.0.1:5000Allow camera access.Input your data (Sleep hours, Message count, etc.).Click "Capture & Identify Stress".📊 Data MappingThe system maps facial emotions into the Stress Model using the following logic:Happy $\rightarrow$ 0Neutral $\rightarrow$ 1Sad $\rightarrow$ 2Angry/Fear/Disgust $\rightarrow$ 3Surprise $\rightarrow$ 4📜 Requirements (requirements.txt)Plaintextflask
opencv-python
tensorflow
scikit-learn
joblib
pandas
numpy
Pillow
🤝 ContributingFor new contributors (newcomers):Fork the Project.Create your Feature Branch (git checkout -b feature/NewAI).Commit your Changes (git commit -m 'Add some NewAI').Push to the Branch (git push origin feature/NewAI).Open a Pull Request.
