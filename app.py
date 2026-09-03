import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Factory QA Dashboard",
    page_icon="⚙️",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Factory QA Pipeline")
    st.markdown(
        """
        This application uses a trained CNN model to inspect cast metal
        impellers and identify visible manufacturing defects.
        """
    )
    st.divider()
    st.write("**Developed by:** LAKSHMI DEEPAK P")

# Main dashboard
st.title("Automated Visual Quality Control")
st.caption("AI-powered casting defect inspection dashboard")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("casting_defect_model.keras")


model = load_model()

uploaded_file = st.file_uploader(
    "Upload a casting image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Preprocess image
    grayscale_image = image.convert("L").resize((300, 300))
    image_array = np.array(grayscale_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=(0, -1))

    # Prediction
    with st.spinner("Running AI inspection..."):
        prediction = model.predict(image_array, verbose=0)

    # Class 0 = def_front, Class 1 = ok_front
    ok_score = float(prediction[0][0])
    defect_score = 1.0 - ok_score

    if ok_score >= 0.5:
        result = "PASS"
        confidence = ok_score
        result_color = "success"
    else:
        result = "FAIL"
        confidence = defect_score
        result_color = "error"

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Inspected Component")
        st.image(
            image,
            caption="Uploaded Factory Part",
            use_container_width=True,
        )

    with right_column:
        st.subheader("Inspection Result")

        if result_color == "success":
            st.success("✅ PASS — Part classified as OK")
        else:
            st.error("🚨 FAIL — Defect detected")

        st.metric(
            label="Model Confidence",
            value=f"{confidence * 100:.2f}%",
        )

        st.progress(confidence)

        st.write(f"**OK probability:** {ok_score * 100:.2f}%")
        st.write(f"**Defect probability:** {defect_score * 100:.2f}%")

# Technical details
with st.expander("Technical Details"):
    st.markdown(
        """
        The application uses a Convolutional Neural Network (CNN) to extract
        visual features from the grayscale casting image.

        The image is resized to **300 × 300 pixels** before inference.
        Convolutional layers identify patterns such as cracks, surface damage,
        and casting irregularities.

        **GlobalAveragePooling2D** reduces each extracted feature map to a
        single value before classification. This decreases the number of
        trainable parameters, reduces overfitting, and makes the model more
        computationally efficient than using a large fully connected layer.
        """
    )
