import streamlit as st
import numpy as np
import os

import torch
# import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Install required packages
st.write("Welcome")
st.write("Installing required packages...")

# Assuming CUDA is available, install the GPU version of PyTorch
os.system("pip install torch torchvision")

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(image).unsqueeze(0)
    return img

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
        img_tensor = preprocess_image(image)

        # Load the PyTorch model
        st.write("Loading the model...")
        model_path = 'Trilokesh_Weather_Model.pth'
        if not os.path.exists(model_path):
            st.write("Downloading the model file...")
            # Assuming the model file is available for download
            # You need to specify the URL or the method of downloading the model file
            # For simplicity, this code assumes the model file is already available
            pass
        loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
        loaded_model.eval()

        # Make prediction
        st.write("Making prediction...")
        with torch.no_grad():
            prediction = loaded_model(img_tensor)
            predicted_class_index = torch.argmax(prediction).item()
            predicted_class = classes[predicted_class_index]

        st.write("Prediction:", predicted_class)

    # Add animation using mathematical function
    st.subheader("Animation using a mathematical function")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    st.line_chart(np.column_stack((x, y)), use_container_width=True)

if __name__ == '__main__':
    main()
