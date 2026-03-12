import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Load model (cached so it loads only once) ─────────────────────────────────
@st.cache_resource
def load_age_gender_model():
    return load_model("Age_Sex_Detection.h5")

model = load_age_gender_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Age & Gender Detector")
st.markdown("Upload a face image and the model will predict the **age** and **gender**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")   # ensure 3-channel RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🚀 Detect Age & Gender"):
        with st.spinner("Analysing..."):
            # ── Pre-process (mirrors the original gui.py logic) ───────────────
            img_resized = image.resize((48, 48))
            img_array  = np.array(img_resized)          # shape (48, 48, 3)
            img_array  = np.array([img_array]) / 255.0  # shape (1, 48, 48, 3)

            # ── Predict ───────────────────────────────────────────────────────
            pred   = model.predict(img_array)
            age    = int(np.round(pred[0][0]))   # regression output
            sex    = int(np.round(pred[1][0]))   # 0 = Male, 1 = Female
            gender = ["Male", "Female"][sex]

        # ── Display results ───────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        col1.metric("🎂 Predicted Age",    str(age) + " yrs")
        col2.metric("⚧  Predicted Gender", gender)
