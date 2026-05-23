import json
import base64
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/plant_model_fixed.keras"
BACKGROUND_PATH = "assets/backgroundd.jpg"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="GreenMind Vision",
    page_icon="🌿",
    layout="wide"
)

def get_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()

background = get_base64(BACKGROUND_PATH)

st.markdown(f"""
<style>
.stApp {{
    background-image:
        linear-gradient(rgba(0,0,0,0.62), rgba(0,0,0,0.62)),
        url("data:image/jpg;base64,{background}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.block-container {{
    max-width: 1200px;
    padding-top: 2rem;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

#MainMenu, footer {{
    visibility: hidden;
}}

h1, h2, h3, label, p, span {{
    color: white !important;
}}

.main-title {{
    text-align: center;
    font-size: 56px;
    font-weight: 900;
    margin-bottom: 8px;
    color: white;
}}

.subtitle {{
    text-align: center;
    font-size: 20px;
    color: #eeeeee;
    margin-bottom: 50px;
}}

.section-title {{
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin-bottom: 20px;
}}

[data-testid="stFileUploader"] {{
    background: rgba(15,15,20,0.65);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.22);
    padding: 20px;
}}

[data-testid="stImage"] img {{
    border-radius: 20px;
}}

[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    padding: 22px;
    backdrop-filter: blur(14px);
}}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {{
    color: white !important;
}}

.stAlert {{
    border-radius: 18px;
}}

.project-box {{
    margin-top: 40px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 24px;
    padding: 30px;
    backdrop-filter: blur(14px);
    color: white;
    font-size: 18px;
    line-height: 1.8;
}}

.project-title {{
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 15px;
}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

model = load_model()
class_names = load_json("class_names.json")
recommendations = load_json("recommendations.json")

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]

    index = int(np.argmax(predictions))
    predicted_class = class_names[index]
    confidence = float(predictions[index] * 100)

    recommendation = recommendations.get(
        predicted_class,
        "No recommendation available."
    )

    return predicted_class, confidence, recommendation

st.markdown(
    '<div class="main-title">🌿 GreenMind Vision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered plant disease classification using deep learning and recommendation generation.</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        '<div class="section-title">Upload Plant Leaf Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG, WEBP",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )

with right:
    st.markdown(
        '<div class="section-title">Prediction Results</div>',
        unsafe_allow_html=True
    )

    if uploaded_file:
        with st.spinner("Analyzing image..."):
            predicted_class, confidence, recommendation = predict_image(image)

        clean_label = predicted_class.replace("___", " - ").replace("_", " ")

        st.metric("Detected Condition", clean_label)
        st.metric("Confidence Score", f"{confidence:.2f}%")
        st.info(recommendation)

    else:
        st.warning("Please upload a plant leaf image to begin prediction.")

st.markdown("""
<div class="project-box">
    <div class="project-title">Project Connection</div>
    This demo represents the deployment stage of the Smart Plant Health Detection System.
    The system uses the trained EfficientNetB0 deep learning model to classify healthy and diseased plant leaves,
    generate confidence scores, and provide recommendation messages based on detected plant conditions.
</div>
""", unsafe_allow_html=True)