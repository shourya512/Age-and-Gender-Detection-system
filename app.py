import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="🔍",
    layout="centered",
)

@st.cache_resource
def load_age_gender_model():
    return load_model("Age_Sex_Detection.h5")

model = load_age_gender_model()

st.title("🔍 Age & Gender Detector")
st.markdown("Upload a face image and the model will predict the **age** and **gender**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=700)

    if st.button("🚀 Detect Age & Gender"):
        with st.spinner("Analysing..."):
            img_resized = image.resize((48, 48))
            img_array = np.array([np.array(img_resized)]) / 255.0
            pred = model.predict(img_array)
            sex_raw = float(pred[0][0][0])
            age = int(np.round(pred[1][0][0]))
            sex = 1 if sex_raw >= 0.5 else 0
            gender = ["Male", "Female"][sex]

        col1, col2 = st.columns(2)
        col1.metric("🎂 Predicted Age", str(age) + " yrs")
        col2.metric("⚧  Predicted Gender", gender)