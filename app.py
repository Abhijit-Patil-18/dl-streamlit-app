import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("model.h5")

# Class names (IMPORTANT: same order as training folders)
class_names = ['cats', 'dogs']

st.title("🐶🐱 Image Classifier")
st.write("Upload an image (Cat or Dog)")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")