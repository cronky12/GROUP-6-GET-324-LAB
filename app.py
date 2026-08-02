import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ============================================================
# CONFIG — these MUST match what you used during training.
# Check the notebook's IMG_SIZE / preprocessing cell before
# trusting this app's predictions. Mismatch here silently
# produces wrong predictions with no error thrown.
# ============================================================
HF_REPO_ID = "Abasiofon001/Concrete_Crack_Screening"  
MODEL_FILENAME = "potato_blight_best_model.keras"       
IMG_SIZE = (224, 224)                                   
CLASS_NAMES = ["Healthy Potato", "Potato Early Blight"]  # index 0 -> label 0, index 1 -> label 1
DECISION_THRESHOLD = 0.5

st.set_page_config(page_title="Potato Leaf Classifier", page_icon="🥔", layout="centered")


@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, pil_img: Image.Image):
    x = preprocess_image(pil_img)
    prob = float(model.predict(x, verbose=0).ravel()[0])
    label_idx = int(prob >= DECISION_THRESHOLD)
    return CLASS_NAMES[label_idx], prob


st.title("🥔 Potato Leaf Classifier")
st.subheader("Healthy vs Early Blight")
st.caption(
    "Trained on the PlantVillage potato subset (lab-condition leaf photos). "
    "This model only distinguishes Healthy from Early Blight — it was not trained on "
    "Late Blight or any other crop's disease, and will misclassify anything outside "
    "those two categories with unwarranted confidence."
)

with st.spinner("Loading model..."):
    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"Failed to load model from {HF_REPO_ID}: {e}")
        st.info(
            "wrong MODEL_FILENAME (check the exact filename in your HF repo's Files tab), "
            "the repo is private and needs a token, or the repo hasn't finished uploading yet."
        )

if model_loaded:
    uploaded_file = st.file_uploader(
        "Upload a potato leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running inference..."):
            label, prob = predict(model, pil_img)

        st.subheader("Result")
        if label == "Potato Early Blight":
            st.error(f"**{label}** — predicted probability: {prob:.1%}")
        else:
            st.success(f"**{label}** — predicted probability of blight: {prob:.1%}")

        confidence = prob if label == "Potato Early Blight" else 1 - prob
        st.progress(confidence)
        st.caption(f"Model confidence in this prediction: {confidence:.1%}")

        with st.expander("Raw output"):
            st.json({"raw_sigmoid_output": prob, "predicted_class": label, "threshold": DECISION_THRESHOLD})
    else:
        st.info("Upload an image to get a prediction.")

st.divider()
st.caption(
    "⚠️ Binary Healthy-vs-Early-Blight only — Late Blight and other diseases were excluded "
    "from training and will be misclassified. Trained on controlled-background lab photos "
    "(PlantVillage), not validated on field photos with natural backgrounds, varying lighting, "
    "or phone-camera blur. This is a portfolio/demo model, not an agricultural diagnostic tool."
)
