import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match model's expected sizing
    img_array = np.array(img)  # Convert PIL image to numpy array
    if img_array.shape[-1] == 4:  # Check if the image has an alpha channel
        img_array = img_array[..., :3]  # Drop the alpha channel
    img_array = img_array / 255.0  # Normalize pixel values to between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the classes (assuming 11 classes for demonstration)
classes = ['dew', 'fog/smog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

# Load the pretrained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"
model = tf.keras.Sequential([hub.KerasLayer(model_url, input_shape=(224, 224, 3))])

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
            try:
                # Preprocess the image
                img_array = preprocess_image(image)

                # Make prediction
                st.write("Making prediction...")
                prediction = model.predict(img_array)
                predicted_class_idx = np.argmax(prediction[0])  # Get the index of the highest probability
                predicted_class = classes[predicted_class_idx]

                st.write("Prediction:", predicted_class)
            except Exception as e:
                st.write(f"Error in making prediction: {e}")

    # Add animation using a mathematical function
    st.subheader("Animation using a mathematical function")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    st.line_chart(np.column_stack((x, y)), use_container_width=True)

if __name__ == '__main__':
    main()
