# Age and Gender Detection System

A machine learning-powered web application that detects **age** and **gender** from facial images using a deep learning model trained on the UTKFace dataset. Built with TensorFlow/Keras and deployed via Streamlit on Hugging Face Spaces.

---

## 🚀 Live Demo

👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/shourya512/age-gender-detection)

---

## 📌 Features

- Predicts **age** (regression) and **gender** (classification) from a single uploaded image
- Trained on **20,000+ labeled face images** from the UTKFace dataset
- Clean and simple web interface — no coding knowledge required
- Real-time predictions with instant results

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model (CNN) |
| NumPy | Numerical computations and array manipulation |
| Pillow (PIL) | Image loading and preprocessing |
| Streamlit | Interactive web application UI |
| Hugging Face Spaces | Deployment platform |
| Jupyter Notebook | Model training and experimentation |

---

## 🧠 How It Works

1. **Model Training** — A Convolutional Neural Network (CNN) was trained on the UTKFace dataset using Keras. The model has two output heads:
   - Age prediction (regression)
   - Gender prediction (binary classification — Male/Female)

2. **Preprocessing** — Uploaded images are converted to RGB, resized to 48×48 pixels, and normalized (pixel values scaled to 0–1).

3. **Prediction** — The preprocessed image is passed through the trained model (`Age_Sex_Detection.h5`), which returns the predicted age and gender.

4. **Web App** — Results are displayed instantly on a Streamlit interface.

---

## 📁 Project Structure

```
Age-and-Gender-Detection-system/
│
├── age_gender_detector.ipynb   # Model training notebook
├── app.py                      # Streamlit web application
├── gui.py                      # Alternate GUI version
├── Age_Sex_Detection.h5        # Saved trained model
├── UTKFaceages.npy             # Age labels from UTKFace dataset
├── UTKFacegenders.npy          # Gender labels from UTKFace dataset
└── requirements.txt            # Project dependencies
```

---

## ⚙️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/shourya512/Age-and-Gender-Detection-system.git
cd Age-and-Gender-Detection-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow
streamlit
numpy
Pillow
```

---

## 📊 Dataset

**UTKFace** — A large-scale face dataset with annotations for age (0–116), gender (Male/Female), and ethnicity.

- 20,000+ images
- Diverse age range and ethnicities
- Publicly available for research use

---

## 📈 Model Evaluation

| Task | Type | Metric |
|---|---|---|
| Gender Prediction | Classification | Accuracy |
| Age Prediction | Regression | Mean Absolute Error (MAE) |

---

## 🌐 Deployment

The app was initially built for **Streamlit Cloud** but was deployed on **Hugging Face Spaces** due to memory and dependency constraints with TensorFlow on Streamlit's free tier. Hugging Face Spaces provides better support for ML-heavy applications.

---

## 🙋 Author

**Shourya** — [GitHub](https://github.com/shourya512)
