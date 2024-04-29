import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import gdown

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match model's expected sizing
    img_array = np.array(img)  # Convert PIL image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values to between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the classes
classes = ['dew', 'fog/smog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Download the model configuration file
model_config_url = 'https://drive.google.com/uc?id=13eVU4Mgnjyw44FQXFND6bB2PR5I0gYUc'
model_config_path = 'model_config.json'
gdown.download(model_config_url, model_config_path, quiet=False)

# Load the model configuration from the downloaded file
with open(model_config_path, "r") as json_file:
    model_config = json.load(json_file)

# Create a new model using the loaded configuration
model = tf.keras.models.model_from_json(model_config)

# Load the pre-trained weights
model.load_weights("Trilokesh_Weather_Model.h5")

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
