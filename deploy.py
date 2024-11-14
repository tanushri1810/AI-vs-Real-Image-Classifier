import streamlit as st
import numpy as np
import cv2 as cv
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("Model.h5")

img_size = 48
categories = ['Real', 'AI Generated']

def preprocess_image(image):
    image = image.convert("RGB")
    img_array = np.array(image)
    img_array = cv.resize(img_array, (img_size, img_size))
    img_array = img_array / 255.0  
    img_array = img_array.reshape(-1, img_size, img_size, 3)  
    return img_array

def make_prediction(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return categories[int(prediction[0] <= 0.5)]  

st.title("AI vs. Real Image Classifier")
st.write("Upload an image to determine if it is AI Generated or Real.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label = make_prediction(image)
    st.write(f"The given image is **{label}**.")

    