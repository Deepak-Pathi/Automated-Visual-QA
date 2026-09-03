import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Factory QA Pipeline", layout="centered")
st.title("⚙️ Automated Visual Quality Control")
st.write("Upload an image of a casted metal impeller to instantly detect manufacturing defects.")

# 2. Load the trained model (Cached so it doesn't reload on every button click)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('casting_defect_model.keras')

model = load_model()

# 3. Create the File Uploader
uploaded_file = st.file_uploader("Drop a casting image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Factory Part", use_container_width=True)

    # 4. Preprocess the image to match our training data
    # Convert to grayscale, resize to 300x300, convert to array, and expand dimensions
    img = image.convert("L") 
    img = img.resize((300, 300))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Shape becomes (1, 300, 300, 1)

    # 5. Make the Prediction
    st.write("Running inspection...")
    prediction = model.predict(img_array)
    
    # 6. Display the Results
    st.divider()
    # Remember: 0 is Ok (def_front is not detected), 1 is Defective (depending on how classes were sorted)
    # tf.keras.utils.image_dataset_from_directory sorts folders alphabetically: 'def_front' is 0, 'ok_front' is 1.
    
    # Let's dynamically check the score. Since 'def_front' is class 0 and 'ok_front' is class 1:
    score = prediction[0][0]
    
    if score > 0.5:
        st.success(f"✅ **PASS**: This part is OK. (Confidence: {score*100:.2f}%)")
    else:
        st.error(f"🚨 **FAIL**: Defect Detected! (Confidence: {(1-score)*100:.2f}%)")