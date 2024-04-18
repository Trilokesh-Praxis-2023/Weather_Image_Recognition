import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import random

# Load the Keras model
loaded_model = tf.keras.models.load_model('Trilokesh_Weather_Model.h5', compile=False)

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match model's expected sizing
    img_array = np.array(img)  # Convert PIL image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values to between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the classes
classes = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']


# Streamlit app
def main():
    st.title("Weather Image Classifier")
    st.write("This app classifies weather images into categories: dew, fog/smog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, snow")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(image)

        # Make prediction
        prediction = loaded_model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]

        st.write("Prediction:", predicted_class)



# Add some styling
st.markdown(
"""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #d1d8e0;
    }
    .Widget>label {
        color: #111;
        font-weight: bold;
    }
    .st-bqplot svg {
        color: #111;
    }
    .st-bqplot .bqplot > g > g > text {
        fill: #111;
    }
</style>
""",
unsafe_allow_html=True
)

if __name__ == '__main__':
    main()
