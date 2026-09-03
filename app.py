import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CastGuard | Factory QA",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional enterprise styling
st.markdown(
    """
    <style>
        .stApp {
            background: #0b1220;
            color: #e5e7eb;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
            border-right: 1px solid #263449;
        }

        [data-testid="stSidebar"] * {
            color: #dbeafe;
        }

        .hero {
            padding: 28px 32px;
            border-radius: 18px;
            background: linear-gradient(135deg, #172554, #0f766e);
            border: 1px solid #2563eb;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
            margin-bottom: 25px;
        }

        .hero h1 {
            color: white;
            margin: 0;
            font-size: 2.4rem;
        }

        .hero p {
            color: #bfdbfe;
            margin: 8px 0 0;
            font-size: 1.05rem;
        }

        .section-title {
            color: #93c5fd;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 12px 0;
        }

        .result-card {
            padding: 24px;
            border-radius: 16px;
            background: #111827;
            border: 1px solid #263449;
            min-height: 270px;
        }

        .pass-badge,
        .fail-badge {
            display: inline-block;
            padding: 8px 18px;
            border-radius: 999px;
            font-weight: 800;
            letter-spacing: 1px;
            margin: 8px 0 20px;
        }

        .pass-badge {
            color: #bbf7d0;
            background: #14532d;
            border: 1px solid #22c55e;
        }

        .fail-badge {
            color: #fecaca;
            background: #7f1d1d;
            border: 1px solid #ef4444;
        }

        [data-testid="stMetric"] {
            background: #172033;
            border: 1px solid #334155;
            padding: 16px;
            border-radius: 12px;
        }

        [data-testid="stFileUploader"] {
            background: #111827;
            border: 1px dashed #3b82f6;
            border-radius: 14px;
            padding: 8px;
        }

        .footer-note {
            color: #64748b;
            text-align: center;
            font-size: 0.85rem;
            margin-top: 35px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ CastGuard QA")
    st.caption("AI-powered manufacturing intelligence")
    st.divider()

    st.markdown("### Project Overview")
    st.write(
        "CastGuard uses a trained Convolutional Neural Network to inspect "
        "cast metal impellers and identify visible manufacturing defects."
    )

    st.divider()
    st.markdown("### Project Information")
    st.write("**Inspection type:** Visual defect detection")
    st.write("**Input format:** JPG, JPEG, PNG")
    st.write("**Image size:** 300 × 300 pixels")

    st.divider()
    st.write("**Developed by:** LAKSHMI DEEPAK P")
    st.caption("Factory Quality Assurance Dashboard")

# Header
st.markdown(
    """
    <div class="hero">
        <h1>Automated Visual Quality Control</h1>
        <p>Industrial casting inspection powered by computer vision</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Upload Inspection Image</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a casting image for AI inspection",
    type=["jpg", "jpeg", "png"],
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("casting_defect_model.keras")


if uploaded_file is not None:
    model = load_model()
    image = Image.open(uploaded_file)

    grayscale_image = image.convert("L").resize((300, 300))
    image_array = np.array(grayscale_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=(0, -1))

    with st.spinner("Analyzing component with AI..."):
        prediction = model.predict(image_array, verbose=0)

    # Binary classifier: output represents OK probability
    ok_score = float(prediction[0][0])
    defect_score = 1.0 - ok_score

    if ok_score >= 0.5:
        result = "PASS"
        confidence = ok_score
    else:
        result = "FAIL"
        confidence = defect_score

    left_column, right_column = st.columns([1.05, 1], gap="large")

    with left_column:
        st.markdown(
            '<div class="section-title">Inspected Component</div>',
            unsafe_allow_html=True,
        )
        st.image(
            image,
            caption="Uploaded Factory Part",
            use_container_width=True,
        )

    with right_column:
        st.markdown(
            '<div class="section-title">Inspection Result</div>',
            unsafe_allow_html=True,
        )

        badge_class = "pass-badge" if result == "PASS" else "fail-badge"
        badge_text = (
            "✓ PASS — COMPONENT APPROVED"
            if result == "PASS"
            else "⚠ FAIL — DEFECT DETECTED"
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="{badge_class}">{badge_text}</div>
                <p>AI classification completed successfully.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            label="Model Confidence",
            value=f"{confidence * 100:.2f}%",
        )

        st.progress(confidence)

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:
            st.metric("OK Probability", f"{ok_score * 100:.2f}%")

        with probability_col2:
            st.metric("Defect Probability", f"{defect_score * 100:.2f}%")

# Technical details
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Technical Details"):
    st.markdown(
        """
        This application uses a **Convolutional Neural Network (CNN)** to
        extract visual features from grayscale casting images.

        Images are resized to **300 × 300 pixels** before inference.
        Convolutional layers identify patterns such as cracks, surface damage,
        and casting irregularities.

        **GlobalAveragePooling2D** converts each feature map into a single
        representative value. This reduces trainable parameters, limits
        overfitting, and provides a more efficient alternative to a large
        fully connected layer.
        """
    )

st.markdown(
    '<div class="footer-note">CastGuard QA • Intelligent Manufacturing Inspection</div>',
    unsafe_allow_html=True,
)
