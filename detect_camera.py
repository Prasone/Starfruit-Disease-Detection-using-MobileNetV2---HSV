import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("model/best_model.h5")

classes = [
    "Anthracnose Disease",
    "Bed Bugs Disease",
    "Fruit Borer Disease",
    "Healthy Fruits",
    "Healthy Leaf"
]

IMG_SIZE = 224

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    label = f"{classes[class_index]} ({confidence:.2f})"

    cv2.putText(
        frame,
        label,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Belimbing Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()