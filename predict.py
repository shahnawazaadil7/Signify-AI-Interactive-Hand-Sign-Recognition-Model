import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the DNN model and label encoder from pickle
with open('./dnn_model.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
label_encoder = model_dict['label_encoder']

# Setup MediaPipe hands detection
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {idx: label for idx, label in enumerate(label_encoder.classes_)}
confidence_threshold = 0.4

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image from webcam. Exiting...")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    all_x = []
    all_y = []
    hand_presence = 0  # Default to no hands

    if results.multi_hand_landmarks:
        hand_presence = len(results.multi_hand_landmarks)  # Set 1 or 2 based on the number of detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))]
            y_ = [hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))]
            all_x.extend(x_)
            all_y.extend(y_)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Add normalized hand landmark positions
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure we have 84 features
        if len(data_aux) == 42:
            data_aux.extend([0] * 42)  # Pad for the second hand
        elif len(data_aux) != 84:
            print(f"Feature length mismatch: {len(data_aux)}")
            continue  # Skip this frame if features are not correct

        # Predict using the DNN model
        try:
            probs = model.predict(np.asarray([data_aux]))[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            continue

        predicted_label_index = np.argmax(probs)
        predicted_label = labels_dict[predicted_label_index]
        max_prob = np.max(probs)

        # Display predicted character and emoji
        if max_prob >= confidence_threshold:
            display_text = f"{predicted_label} ({max_prob:.1f})"
        else:
            display_text = f"Unknown ({max_prob:.1f})"

        # Draw a bounding box around the hand with prediction text
        if all_x and all_y:
            x1 = int(min(all_x) * W) - 10
            y1 = int(min(all_y) * H) - 10
            x2 = int(max(all_x) * W) + 10
            y2 = int(max(all_y) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255),
                        3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()