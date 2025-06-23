import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/mask_detector_model.h5")
labels = ["Mask", "No Mask"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for model
    resized = cv2.resize(frame, (100, 100))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 100, 100, 3))

    # Make prediction
    result = model.predict(reshaped)
    label = labels[np.argmax(result)]
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    # Draw result on frame
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (20, 20), (300, 100), color, 2)

    cv2.imshow("Face Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
