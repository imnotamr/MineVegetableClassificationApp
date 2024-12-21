# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```


## Vegetable Classification Model

The trained model for the **Vegetable Classification App** is available for download! 🎉

### **Download the Model**
[Click here to download the model file (Vegetable_model_amr.h5)](https://github.com/imnotamr/Vegetable-Classification-App/releases/download/v1.0/Vegetable_model_amr.h5)

### **Usage Instructions**
1. Download the model file from the link above.
2. Place the file in your project directory.
3. Load the model in your Python script:
   ```python
   from tensorflow.keras.models import load_model

   model = load_model('Vegetable_model_amr.h5')
