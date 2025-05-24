🤟 Real-Time Indian Sign Language (ISL) Translator

Smart India Hackathon 2024 Project
Developed using Dense Neural Networks, MediaPipe, and OpenCV

📝 Overview

This project presents a real-time hand sign recognition system designed for Indian Sign Language (ISL) translation. Built with a combination of MediaPipe for hand tracking and TensorFlow/Keras for gesture classification, this system can capture and classify live hand gestures using a webcam.

Developed by a team of 6 CSE-DS students from LIET for the Smart India Hackathon 2024, the project involved collecting and training on 57,600 hand gesture images.

⸻

📂 Folder Structure
```
.
├── data/                   # Folder containing captured images, organized per class
├── models/                 # Folder to store the trained model and preprocessed data
│   ├── dnn_model.p         # Trained Dense Neural Network model
│   ├── label_encoder.p     # Saved LabelEncoder object
│   └── preprocessed_data.p # Pickled data after preprocessing
├── capture_data.py         # Script to collect gesture images via webcam
├── preprocess_data.py      # Script to extract hand landmarks and label data
├── train_model.py          # Script to train the DNN model
├── run_realtime_demo.py    # Real-time prediction using webcam and trained model
└── README.md               # Project documentation

```
⸻

🚀 Features
	•	Captures hand gesture data with webcam
	•	Detects hand landmarks using MediaPipe
	•	Extracts and normalizes 84 hand landmark coordinates (42 for each hand)
	•	Includes hand presence as an additional feature
	•	Trains a Dense Neural Network (DNN) to classify 36 ISL signs
	•	Real-time prediction and feedback using OpenCV and the trained model
	•	Total dataset: 57,600 images (36 classes × 1600 images per class approx.)

⸻

🔧 Setup Instructions

1. Clone the Repository
```
git clone https://github.com/shahnawazaadil7/Signify-AI-Interactive-Hand-Sign-Recognition-Model/
cd Signify-AI-Interactive-Hand-Sign-Recognition-Model
```
2. Install Dependencies
```
pip install opencv-python mediapipe scikit-learn numpy tensorflow
```

⸻

📸 Data Collection

To capture gesture data for training:
```
python capture_data.py
```
•	You will be prompted class-wise to collect 100 samples per gesture.
•	Press Q to start capturing for each class.

⸻

🔍 Preprocessing

To process and extract landmarks:
```
python preprocess_data.py
```
•	Uses MediaPipe to detect hands and extract 84 hand landmark coordinates.
•	Pads data if only one hand is present and includes a hand presence flag.
•	Saves the feature vectors and labels to models/preprocessed_data.p.

⸻

🧠 Training the Model

To train the Dense Neural Network:
```
python train_model.py
```
•	Trains on the processed dataset with an 80/20 train-test split.
•	Uses dropout and ReLU activation to reduce overfitting.
•	Saves the trained model as models/dnn_model.p.

⸻

🖥️ Real-Time Prediction

To run the live demo:
```
python run_realtime_demo.py
```
•	Opens your webcam and predicts ISL signs in real time.
•	Displays bounding boxes and prediction confidence on-screen.

⸻

🧪 Model Architecture
	•	Input Layer: 85 features (84 landmark points + 1 hand presence flag)
	•	Hidden Layers:
	•	Dense(128) → ReLU
	•	Dense(64) → ReLU → Dropout(0.3)
	•	Dense(32) → ReLU → Dropout(0.3)
	•	Output Layer: Softmax over 36 classes

⸻

🏆 Hackathon Details
	•	Event: Smart India Hackathon 2024
	•	Team Size: 6
	•	Track: Accessibility / Sign Language Translation
	•	Dataset Size: 57,600 gesture images
	•	Tools Used: Python, TensorFlow, OpenCV, MediaPipe, scikit-learn

⸻

📌 Future Improvements
	•	Add multi-word sentence formation using LSTM or attention models
	•	Implement hand orientation and depth normalization
	•	Deploy the system as a web or mobile app
	•	Incorporate dynamic gestures (like movement-based signs)

⸻

📃 License

This project is open-sourced under the MIT License.
