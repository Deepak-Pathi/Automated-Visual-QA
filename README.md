# ⚙️ Automated Visual Quality Control Pipeline

## The Business Problem
In manufacturing, manual visual inspection of assembly line parts is slow, expensive, and prone to human error. This project automates the quality assurance process using Computer Vision, instantly classifying casted metal impellers as "Defective" or "OK" to save operational costs and reduce waste.

## The Solution & Tech Stack
An end-to-end Deep Learning pipeline and interactive web dashboard.
* **Deep Learning:** TensorFlow & Keras (Custom Convolutional Neural Network)
* **Web Framework:** Streamlit
* **Language:** Python
* **Deployment:** Streamlit Community Cloud

## Technical Highlight: Model Optimization
Standard CNN architectures using fully connected `Flatten` layers resulted in a massive parameter count (20M+), making web deployment sluggish. By re-architecting the network to use `GlobalAveragePooling2D`, I successfully:
* Shrunk the model file size from ~80MB down to under 1MB.
* Reduced trainable parameters to under 100,000.
* Maintained high validation accuracy while drastically improving inference speed for the web dashboard.

## Author
Lakshmi Deepak P
