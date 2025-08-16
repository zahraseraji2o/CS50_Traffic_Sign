# Traffic Sign Recognition - CS50 AI

This project implements a **Convolutional Neural Network (CNN)** for classifying traffic signs, built as part of the Harvard CS50 AI course.

---

## Features
- Preprocesses images to 30x30 pixels.
- Trains a simple CNN with two convolutional layers.
- Evaluates accuracy on a test set.
- Optionally saves the trained model as `.h5`.

---

## Dataset
This project uses the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.  
Download the dataset [here](https://benchmark.ini.rub.de/gtsrb_dataset.html).

> Note: The dataset is **not included** in this repository due to its size.  
> Users should download and extract it separately.

---

## Usage
```bash
python3 traffic.py <data_directory> [model.h5]
Dependencies

Python 3.x

OpenCV (cv2)

TensorFlow (tensorflow)

scikit-learn (sklearn)

NumPy (numpy)

You can install the Python packages with pip:

pip install opencv-python tensorflow scikit-learn numpy
