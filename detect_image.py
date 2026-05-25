import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("model/best_model.h5")

# =========================
# CLASS
# =========================
classes = [
    "Anthracnose Disease",
    "Bed Bugs Disease",
    "Fruit Borer Disease",
    "Healthy Fruits",
    "Healthy Leaf"
]

IMG_SIZE = 224

# =========================
# IMAGE PATH
# =========================
# image_path = "anthracnose.jpg"
# image_path = "bedbugs.jpg"
image_path = "fruitborer1.jpg"
# image_path = "Healthy Fruit.jpg"
# image_path = "Healthy Leaf.jpg"

# =========================
# READ IMAGE
# =========================
img = cv2.imread(image_path)

if img is None:
    print("ERROR: gambar tidak ditemukan")
    exit()

display = img.copy()

# =========================
# PREPROCESS
# =========================
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# BGR -> HSV
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# normalize
img = img.astype("float32") / 255.0

img = np.expand_dims(img, axis=0)

# =========================
# PREDICT
# =========================
prediction = model.predict(
    img,
    verbose=0
)[0]

# =========================
# ALL CONFIDENCE
# =========================
print("\nHASIL DETEKSI:\n")

for i, class_name in enumerate(classes):

    confidence = prediction[i] * 100

    print(f"{class_name}: {confidence:.2f}%")

# =========================
# BEST RESULT
# =========================
class_index = np.argmax(prediction)

best_confidence = prediction[class_index] * 100

label = f"{classes[class_index]} ({best_confidence:.2f}%)"

print("\nPREDIKSI AKHIR:")
print(label)

# =========================
# DISPLAY
# =========================
cv2.putText(
    display,
    label,
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

cv2.imshow("Starfruit Disease Detection", display)

cv2.waitKey(0)
cv2.destroyAllWindows()