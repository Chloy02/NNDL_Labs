import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Configuration ---
MODEL_PATH = "intel_image_classifier.keras"
IMG_SIZE = (150, 150)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

st.set_page_config(
    page_title="Intel Scene Classifier",
    page_icon="🏞️",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_trained_model():
    """
    Loads the pre-trained Keras model from disk.
    Uses st.cache_resource to load only once.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Make sure the file '{MODEL_PATH}' is in the same directory as app.py")
        return None

model = load_trained_model()

# --- Image Preprocessing ---
def preprocess_image(image_pil):
    """
    Converts a PIL image into the correct format for the model:
    1. Resizes to (150, 150)
    2. Converts to NumPy array
    3. Rescales pixel values (0-1)
    4. Adds a batch dimension
    """
    img = image_pil.resize(IMG_SIZE)
    img_array = np.array(img)
    
    # Handle grayscale images by converting them to 3 channels
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    # Handle images with alpha channel
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    img_array = img_array.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# --- Streamlit UI ---
st.title("🏞️ Intel Scene Classifier")
st.write("Upload an image of a natural scene, and the model will predict its category.")
st.write(f"Categories: {', '.join(CLASS_NAMES)}")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # 2. Preprocess and predict
    if model is not None:
        with st.spinner("Classifying..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            
            # 3. Display the results
            pred_index = np.argmax(prediction)
            pred_class = CLASS_NAMES[pred_index]
            pred_confidence = np.max(prediction) * 100
            
            st.success(f"**Prediction: {pred_class}**")
            st.write(f"Confidence: {pred_confidence:.2f}%")
            
            # Optional: Show confidence for all classes
            st.bar_chart(prediction[0])
            st.subheader("All Class Probabilities:")
            st.dataframe(
                {
                    "Class": CLASS_NAMES, 
                    "Probability": prediction[0]
                }, 
                use_container_width=True
            )
else:
    st.info("Please upload an image file to get started.")