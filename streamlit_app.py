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
    .prediction-history {
        font-size: 16px;
        color: #000;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle (Hero section)
st.markdown('<div class="title"> Welcome to Vegetable Classification App </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of a vegetable to find out what it is!</div>', unsafe_allow_html=True)

# Function to load the trained model dynamically
@st.cache_resource
def load_trained_model():
    model_url = "https://github.com/imnotamr/VegetableClassificationAppFEHU/releases/download/v2.0"
    model_path = "Vegetable_model_fully_compatible.h5"

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

    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    return model

# Load the model
model = load_trained_model()

# Placeholder for prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# File uploader widget
uploaded_file = st.file_uploader("Choose a vegetable image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Class Labels
    class_labels = [
        'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
        'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
        'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
    ]

    # Preprocess the uploaded image
    img = load_img(uploaded_file, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    # Save prediction to session state
    st.session_state.prediction_history.append(predicted_class)

    # Display the predicted class
    st.markdown(f'<div class="prediction-text">Predicted Vegetable: {predicted_class}</div>', unsafe_allow_html=True)

    # Display confidence scores
    confidence_df = pd.DataFrame({
        'Class': class_labels,
        'Confidence': predictions[0]
    }).sort_values(by='Confidence', ascending=True)

    fig = px.bar(
        confidence_df,
        x='Confidence',
        y='Class',
        orientation='h',
        title="Confidence Scores for Each Class",
        labels={'Confidence': 'Confidence (%)', 'Class': 'Vegetable Class'},
        text=confidence_df['Confidence'].apply(lambda x: f'{x:.2%}')
    )
    fig.update_traces(marker_color='#2E86C1', textposition='outside')
    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis_title="Vegetable Class",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

# Add a section for camera input
st.markdown('<div class="subtitle">Or use your camera to capture a vegetable image:</div>', unsafe_allow_html=True)

# Camera input widget
camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    st.image(camera_image, caption="Captured Image", use_container_width=True)

    # Class Labels
    class_labels = [
        'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
        'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
        'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
    ]

    # Preprocess the captured image
    img = load_img(camera_image, target_size=(150, 150))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    # Save prediction to session state
    st.session_state.prediction_history.append(predicted_class)

    # Display the predicted class
    st.markdown(f'<div class="prediction-text">Predicted Vegetable: {predicted_class}</div>', unsafe_allow_html=True)

    # Display confidence scores
    confidence_df = pd.DataFrame({
        'Class': class_labels,
        'Confidence': predictions[0]
    }).sort_values(by='Confidence', ascending=True)

    fig = px.bar(
        confidence_df,
        x='Confidence',
        y='Class',
        orientation='h',
        title="Confidence Scores for Each Class",
        labels={'Confidence': 'Confidence (%)', 'Class': 'Vegetable Class'},
        text=confidence_df['Confidence'].apply(lambda x: f'{x:.2%}')
    )
    fig.update_traces(marker_color='#2E86C1', textposition='outside')
    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis_title="Vegetable Class",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

# Display prediction history
if st.session_state.prediction_history:
    st.markdown('<div class="subtitle">Prediction History:</div>', unsafe_allow_html=True)
    for i, pred in enumerate(st.session_state.prediction_history, 1):
        st.markdown(f'<div class="prediction-history">{i}. {pred}</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Vegetable Classifier App | By Amr Ahmed, Mohamed Yasser, Omar Khaled, Ibrahim Mahmoud </div>', unsafe_allow_html=True)
