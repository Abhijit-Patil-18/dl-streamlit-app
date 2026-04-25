import streamlit as st
import numpy as np
from PIL import Image

st.title("🐶🐱 Image Classifier (Demo)")
st.write("Upload an image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Fake prediction (for deployment demo)
    st.success("Prediction: dogs")
