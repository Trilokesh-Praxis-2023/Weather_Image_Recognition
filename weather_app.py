import streamlit as st

try:
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    import os

    # Install gdown package
    os.system("pip install gdown")

    # Download the model file from Google Drive
    os.system("gdown --id 1H2leUE3T_XOgpJ6j-LMeRZvmj25xbzqA -O Trilokesh_Weather_Model.h5")

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

    # Function to display random images
    def display_random_images(num_images=3):
        st.subheader("Random Images")
        image_folder = "random_images"
        image_files = os.listdir(image_folder)
        random.shuffle(image_files)
        for i in range(num_images):
            image_path = os.path.join(image_folder, image_files[i])
            image = Image.open(image_path)
            st.image(image, caption=f"Random Image {i+1}", use_column_width=True)

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

        # Display random images
        display_random_images()

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

except ImportError:
    st.error("TensorFlow is not available. Please make sure TensorFlow is installed in your environment.")
