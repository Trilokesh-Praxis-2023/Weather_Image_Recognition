import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #6495ED; /* Blue background color */
        padding: 20px;
        border-radius: 10px;
        color: black; /* Text color */
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #888888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
try:
    model = tf.keras.Sequential([hub.KerasLayer(model_url, input_shape=(224, 224, 3))])
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Streamlit app
def main():
    st.sidebar.title("Weather Image Classifier")
    st.sidebar.write("This app classifies weather images into various categories.")

    st.sidebar.header("Upload an Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    st.sidebar.header("Options")
    show_animation = st.sidebar.checkbox("Show Mathematical Function Animation", value=True)

    st.title("Weather Image Classifier")
    st.write("This app classifies weather images into categories: dew, fog/smog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, snow")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True, clamp=True)

        # Button for making prediction
        if st.button('Make Prediction'):
            if model_loaded:
                with st.spinner('Making prediction...'):
                    try:
                        # Preprocess the image
                        img_array = preprocess_image(image)

                        # Make prediction
                        prediction = model.predict(img_array)
                        predicted_class_idx = np.argmax(prediction[0])  # Get the index of the highest probability
                        predicted_class = classes[predicted_class_idx]

                        st.success(f"Prediction: {predicted_class}")
                    except Exception as e:
                        st.error(f"Error in making prediction: {e}")
            else:
                st.error("Model could not be loaded. Prediction cannot be made.")

    if show_animation:
        st.subheader("Animation using a Mathematical Function")
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        st.line_chart(np.column_stack((x, y)), use_container_width=True)

    st.markdown('<div class="footer">Developed with Streamlit</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
