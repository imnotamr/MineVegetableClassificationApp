import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import os
import tensorflow as tf

# Set up Streamlit app with custom HTML and CSS
st.set_page_config(page_title="Vegetables Classification", layout="wide")

# Custom CSS for styling (Background image and text styles)
st.markdown("""
    <style>
    body {
        background-image: url('https://www.toptal.com/designers/subtlepatterns/patterns/ep_naturalblack.png'); 
        background-size: cover;
    }
    .title {
        font-size: 50px;
        font-weight: bold;
        color: #00008B;
        text-align: center;
        margin-top: 50px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
    }
    .subtitle {
        font-size: 20px;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    .footer {
        font-size: 14px;
        color: #808080;
        text-align: center;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle (Hero section)
st.markdown('<div class="title"> Welcome to Vegetable Classification App </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of a vegetable to find out what it is!</div>', unsafe_allow_html=True)

# Function to load the trained model dynamically
@st.cache_resource
def load_trained_model():
    st.write(f"TensorFlow Version: {tf.__version__}")

    model_url = "https://github.com/imnotamr/Vegetable-Classification-App/releases/download/v1.0/Vegetable_model_fully_compatible.h5"
    model_path = "Vegetable_model_fully_compatible.h5"

    # Check if the model file exists locally; if not, download it
    if not os.path.exists(model_path):
        st.info("Downloading the model file...")
        with open(model_path, "wb") as f:
            response = requests.get(model_url)
            if response.status_code == 200:
                f.write(response.content)
                st.success("Model downloaded successfully!")
            else:
                st.error("Failed to download the model. Please check the URL.")
                st.stop()

    # Load the model
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    return model

# Load the model
model = load_trained_model()

# Vegetable info dictionary
vegetable_info = {
    'Bean': 'Beans are rich in protein and fiber.',
    'Bitter_Gourd': 'Bitter gourds are known for their medicinal properties.',
    'Bottle_Gourd': 'Bottle gourds are low in calories and rich in nutrients.',
    'Brinjal': 'Brinjals are high in antioxidants.',
    'Broccoli': 'Broccoli is packed with vitamins and minerals.',
    'Cabbage': 'Cabbages are a great source of dietary fiber.',
    'Capsicum': 'Capsicums are rich in Vitamin C.',
    'Carrot': 'Carrots are excellent for improving vision and skin health.',
    'Cauliflower': 'Cauliflowers are low in carbs and high in fiber.',
    'Cucumber': 'Cucumbers help in hydration and skin health.',
    'Papaya': 'Papayas aid digestion and boost immunity.',
    'Potato': 'Potatoes are versatile and rich in potassium.',
    'Pumpkin': 'Pumpkins are a great source of beta-carotene.',
    'Radish': 'Radishes are excellent for detoxifying the body.',
    'Tomato': 'Tomatoes are rich in Vitamin C and antioxidants.'
}

# Create a section for uploading images
with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a vegetable image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True, output_format="auto")

        # Class Labels
        class_labels = list(vegetable_info.keys())

        # Preprocess the uploaded image
        img = load_img(uploaded_file, target_size=(150, 150))  # Resize to match the input shape of the model
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        # Display the predicted class
        st.markdown(f'<div class="prediction-text">Predicted Vegetable: {predicted_class}</div>', unsafe_allow_html=True)

        # Add vegetable info
        st.markdown(f"**Did You Know?** {vegetable_info.get(predicted_class, 'Information coming soon!')}")

        # Display confidence scores as a table
        confidence_df = pd.DataFrame({
            'Class': class_labels,
            'Confidence': predictions[0]
        }).sort_values(by='Confidence', ascending=False)
        st.write("Confidence Scores (Top 3):")
        st.dataframe(confidence_df.head(3))

        # Add a download button for confidence scores
        st.download_button(
            label="Download Predictions as CSV",
            data=confidence_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
    else:
        st.markdown('<div class="subtitle">Please upload an image to start classifying.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Vegetable Classifier App | By Amr Ahmed, Mohamed Yasser, Omar Khaled, Ibrahim Mahmoud </div>', unsafe_allow_html=True)
