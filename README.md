# WORKSHOP-5-DIP

# 🚗 License Plate Detection using OpenCV

A lightweight computer vision project that detects and crops vehicle license plates from images using OpenCV and a Haar Cascade Classifier.

---

## 📌 Overview

This project implements an end-to-end digital image processing pipeline to detect vehicle registration plates. It applies preprocessing techniques (grayscale conversion and smoothing) to clean the input image, runs a multi-scale Haar feature detector, draws bounding boxes around detected plates, and automatically crops and saves the plate region of interest (ROI).

---

## ⚙️ Features

* **Image Preprocessing:** Grayscale conversion and noise reduction for better detection accuracy.
* **Haar Cascade Classifier:** Fast, multi-scale object detection via `detectMultiScale`.
* **Automatic Cropping:** Extracts and saves detected license plate regions to disk.
* **Side-by-Side Visualization:** Matplotlib display showing the detected plate and the cropped result.

---

## 🛠️ Requirements

Install the required Python packages before running:

## Program--

```python

import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

# ============================================================
# WORKSHOP - 5
# License Plate Detection using OpenCV and Haar Cascade
# ============================================================

# ------------------------------------------------------------
# 1. Download Haar Cascade XML if it is not already available
# ------------------------------------------------------------

cascade_file = "haarcascade_russian_plate_number.xml"

cascade_url = (
    "https://raw.githubusercontent.com/opencv/opencv/"
    "4.x/data/haarcascades/haarcascade_russian_plate_number.xml"
)

if not os.path.exists(cascade_file):
    print("Haar Cascade file not found.")
    print("Downloading Haar Cascade XML...")

    try:
        urllib.request.urlretrieve(cascade_url, cascade_file)
        print("Haar Cascade downloaded successfully!")
    except Exception as e:
        raise RuntimeError(
            "Could not download the Haar Cascade XML file.\n"
            "Please check your internet connection."
        ) from e


# ------------------------------------------------------------
# 2. Load Haar Cascade
# ------------------------------------------------------------

plate_cascade = cv2.CascadeClassifier(cascade_file)

if plate_cascade.empty():
    raise RuntimeError(
        "Haar Cascade could not be loaded.\n"
        "Check the XML file."
    )

print("Haar Cascade loaded successfully!")


# ------------------------------------------------------------
# 3. Load Input Image
# ------------------------------------------------------------

image_file = "carplate.jpeg"

img = cv2.imread(image_file)

if img is None:
    raise FileNotFoundError(
        f"Image '{image_file}' was not found.\n"
        "Make sure carplate.jpeg is in the same folder as this notebook."
    )

print("Image loaded successfully!")


# ------------------------------------------------------------
# 4. Preprocessing
# ------------------------------------------------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# ------------------------------------------------------------
# 5. License Plate Detection
# ------------------------------------------------------------

plates = plate_cascade.detectMultiScale(
    blurred,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 10)
)

print("Number of plates detected:", len(plates))


# ------------------------------------------------------------
# 6. Convert Image for Matplotlib
# ------------------------------------------------------------

img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

cropped_plate = None


# ------------------------------------------------------------
# 7. Draw Bounding Box and Crop Plate
# ------------------------------------------------------------

for (x, y, w, h) in plates:

    # Draw green rectangle
    cv2.rectangle(
        img_rgb,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    # Crop detected plate
    cropped_plate = img_rgb[
        y:y + h,
        x:x + w
    ]

    # Save cropped plate
    cv2.imwrite(
        "plate_crop.png",
        cv2.cvtColor(
            cropped_plate,
            cv2.COLOR_RGB2BGR
        )
    )

    print("License plate cropped and saved as plate_crop.png")


# ------------------------------------------------------------
# 8. Display Output
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

# Original image with detected plate
plt.subplot(1, 2, 1)

if len(plates) > 0:
    plt.title("Detected Plate")
else:
    plt.title("No Plate Detected")

plt.imshow(img_rgb)
plt.axis("off")


# Cropped plate
plt.subplot(1, 2, 2)

if cropped_plate is not None:
    plt.title("Cropped Plate")
    plt.imshow(cropped_plate)
else:
    plt.title("Cropped Plate - Not Found")
    plt.text(
        0.5,
        0.5,
        "No license plate detected",
        ha="center",
        va="center",
        fontsize=12
    )

plt.axis("off")

plt.tight_layout()
plt.show()
```

## Output --


<img width="1379" height="451" alt="image" src="https://github.com/user-attachments/assets/d45f7fba-8ef5-4e4b-b56a-fd53645d28f1" />


## Result --
    Thus, the workshop has been implemented successfully.
