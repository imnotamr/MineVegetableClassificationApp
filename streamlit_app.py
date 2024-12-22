import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model

# Set up Streamlit app with page config
st.set_page_config(page_title="Vegetables Classification", layout="wide")

# Add Dark Mode Toggle
dark_mode = st.sidebar.checkbox("Enable Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: #e5e5e5;
        }
        .title {
            color: #79c0ff;
        }
        .subtitle, .footer {
            color: #58a6ff;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-color: #ffffff;
            color: #000000;
        }
        .title {
            color: #00008B;
        }
        .subtitle, .footer {
            color: #1E90FF;
        }
        </style>
    """, unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title"> Welcome to Vegetable Classification App </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of a vegetable to find out what it is!</div>', unsafe_allow_html=True)

# Load the trained model dynamically
@st.cache_resource
def load_trained_model():
    model_url = "https://github.com/imnotamr/Vegetable-Classification-App/releases/download/v1.0/Vegetable_model_fully_compatible.h5"
    model_path = "Vegetable_model_fully_compatible.h5"

    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            response = requests.get(model_url)
            if response.status_code == 200:
                f.write(response.content)
            else:
                st.error("Failed to download the model. Please check the URL.")
                st.stop()
    return load_model(model_path, compile=False)

# Load the model
model = load_trained_model()

# Upload Section
uploaded_file = st.file_uploader("Choose a vegetable image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Class Labels
    class_labels = [
        'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
        'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
        'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
    ]

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    st.markdown(f"### Predicted Vegetable: **{predicted_class}**")

    # Confidence Table
    confidence_df = pd.DataFrame({
        'Class': class_labels,
        'Confidence': predictions[0]
    }).sort_values(by='Confidence', ascending=False)
    st.dataframe(confidence_df.style.format({'Confidence': '{:.2%}'}).background_gradient(cmap='Blues'))

    # Grad-CAM Explanation
    st.markdown("### Grad-CAM Explanation")
    def grad_cam(model, img_array, layer_name):
        grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, np.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = np.mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap.numpy()

    # Apply Grad-CAM
    heatmap = grad_cam(model, img_array, "conv2d")  # Update layer name based on your model architecture
    heatmap = cv2.resize(heatmap, (150, 150))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img_array[0] * 255.0
    plt.imshow(superimposed_img.astype('uint8'))
    plt.axis('off')
    st.pyplot()

# Footer
st.markdown('<div class="footer">Vegetable Classifier App | By Amr Ahmed, Mohamed Yasser, Omar Khaled, Ibrahim Mahmoud</div>', unsafe_allow_html=True)
