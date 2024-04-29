import streamlit as st
import numpy as np
import os
import urllib.request

import tensorflow as tf
from PIL import Image
from transformers import TFAutoModelForSequenceClassification, AutoConfig, AutoTokenizer

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match model's expected sizing
    img_array = np.array(img)  # Convert PIL image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values to between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the classes
classes = ['dew', 'fog/smog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Streamlit app
def main():
    st.title("Weather Image Classifier")
    st.write("This app classifies weather images into categories: dew, fog/smog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, snow")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Button for making prediction
        if st.button('Make Prediction'):
            # Preprocess the image
            img_array = preprocess_image(image)

            # Load the model configuration
            st.write("Loading the model configuration...")
            model_name = "trilokesh/Weather"  # Use the model identifier from the Hugging Face model hub
            config = AutoConfig.from_pretrained(model_name)
            
            # Load the tokenizer
            st.write("Loading the tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load the model
            st.write("Loading the model...")
            model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)

            # Make prediction
            st.write("Making prediction...")
            prediction = model.predict(img_array)
            predicted_class = classes[np.argmax(prediction)]

            st.write("Prediction:", predicted_class)

    # Add animation using mathematical function
    st.subheader("Animation using a mathematical function")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    st.line_chart(np.column_stack((x, y)), use_container_width=True)

if __name__ == '__main__':
    main()
