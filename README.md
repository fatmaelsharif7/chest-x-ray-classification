# 🩻 Chest X-Ray Classification Using Deep Learning

## 📌 Project Overview

This project uses **Deep Learning and Convolutional Neural Networks (CNNs)** to classify chest X-ray images into two categories: **Normal** and **Abnormal**.

The model processes chest X-ray images and predicts the corresponding class. A user-friendly web application built with Streamlit allows users to upload an X-ray image and view the model's prediction.

The project demonstrates the application of computer vision and deep learning techniques to medical image classification.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and must not be used as a substitute for professional medical advice.

---

## 🎯 Project Objectives

- Build a CNN-based model for chest X-ray image classification.
- Apply image preprocessing techniques using OpenCV and NumPy.
- Evaluate model performance using standard classification metrics.
- Develop an interactive web application using Streamlit.
- Demonstrate an end-to-end deep learning workflow, from model development to inference.

---

## ✨ Features

- **Image Classification:** Predicts whether a chest X-ray image is Normal or Abnormal.
- **Image Preprocessing:** Converts images to grayscale, resizes them, and normalizes pixel values.
- **Deep Learning Model:** Uses a Convolutional Neural Network (CNN).
- **Interactive Web Interface:** Upload an image and display the prediction through Streamlit.
- **Model Evaluation:** Reports accuracy, precision, recall, and test loss.

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **OpenCV**
- **Pillow (PIL)**
- **Streamlit**
- **Jupyter Notebook**

---

## 🧠 Model Architecture

The classification model is built using a Convolutional Neural Network (CNN).

### Architecture Components

- **Input Layer:** Accepts grayscale images with dimensions of `128 × 128 × 1`.
- **Convolutional Layers:** Five convolutional layers with filter sizes of 32, 32, 64, 64, and 128.
- **Batch Normalization:** Helps stabilize the training process.
- **Dropout:** Helps reduce overfitting.
- **Dense Layers:** Fully connected layers with 256 and 128 units.
- **Output Layer:** A single output unit for binary classification.

### Model Parameters

| Parameter | Value |
|---|---:|
| Total Parameters | 8,562,657 |
| Trainable Parameters | 8,561,697 |
| Non-trainable Parameters | 960 |
| Input Image Size | 128 × 128 |
| Input Channels | 1 (Grayscale) |

---

## 🖼️ Image Preprocessing

Before making a prediction, each uploaded image goes through the following preprocessing steps:

1. Convert the image to a NumPy array.
2. Convert the image from RGB to grayscale.
3. Resize the image to `128 × 128` pixels.
4. Convert pixel values to `float32`.
5. Normalize pixel values by dividing them by `255`.
6. Expand the dimensions to match the model's expected input shape: `(1, 128, 128, 1)`.

These steps ensure that uploaded images follow the input format expected by the trained model.

---

## 📊 Model Evaluation

The model was evaluated using a test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | 77.72% |
| Precision | 74.00% |
| Recall | 99.23% |
| Test Loss | 1.0648 |

### Performance Summary

The model achieved a test accuracy of **77.72%** and a recall of **99.23%**.

These results describe the model's performance on the evaluated test dataset. They do not establish clinical reliability or suitability for real-world medical diagnosis.

The interpretation of precision and recall depends on the class encoding used during evaluation.

---

## 🌐 Streamlit Web Application

The project includes an interactive web application developed with Streamlit.

### How It Works

1. The user uploads a chest X-ray image.
2. The application preprocesses the image.
3. The trained CNN model generates a prediction.
4. The application displays the predicted class: Normal or Abnormal.

The application accepts the following image formats:

- JPG
- JPEG
- PNG

---

## 📁 Project Structure

```text
chest-xray-classification/
│
├── APP.py
├── X_Ray.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

### Files Description

| File | Description |
|---|---|
| `APP.py` | Streamlit application for image classification. |
| `X_Ray.ipynb` | Jupyter Notebook containing the model development workflow. |
| `README.md` | Project documentation. |
| `requirements.txt` | Python dependencies required by the project. |
| `.gitignore` | Specifies files and folders excluded from Git tracking. |

**Note:** The trained model file is not included in this repository because of its size.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/fatmaelsharif7/chest-x-ray-classification.git
```

### 2. Navigate to the Project Directory

```bash
cd chest-x-ray-classification
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

The application loads a trained model named `chest_xray_model.h5`.

Because this model file is not included in the repository, you must obtain the trained model separately and place it in the project directory, or update the model path in `APP.py`.

Once the model file is available, run:

```bash
streamlit run APP.py
```

Streamlit will provide a local URL where you can access the application through your browser.

**Important:** The application cannot perform predictions until the trained model file is available at the expected path.

---

## 🚧 Limitations

- The trained model file is not included in this repository.
- The reported metrics reflect evaluation on the available test dataset only.
- Performance may vary on images from different sources or populations.
- The model has not been established as suitable for clinical use.
- Additional validation and testing are needed before any real-world application.

---

## 🔮 Future Improvements

- Improve model performance through hyperparameter tuning.
- Explore transfer learning with pretrained CNN architectures.
- Add confusion matrix and classification report visualizations.
- Evaluate the model on independent datasets.
- Improve model generalization and robustness.
- Deploy the application to a cloud platform.
- Add more detailed prediction explanations and confidence scores.

---

## 👩‍💻 Author

**Fatma Waleed Mohamed Elsharif**

Data Science | Machine Learning | Artificial Intelligence

- **GitHub:** https://github.com/fatmaelsharif7
- **LinkedIn:** https://www.linkedin.com/in/fatma-el-sharif

---

## 📌 Disclaimer

This project is for educational and research purposes only. Predictions generated by the model should not be used to diagnose, treat, or rule out any medical condition. Always consult a qualified healthcare professional for medical evaluation.

---
