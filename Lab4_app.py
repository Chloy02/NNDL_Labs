import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd # Import pandas to create the labeled DataFrame

# --- Configuration ---
MODEL_PATH = "intel_image_classifier.keras"
IMG_SIZE = (150, 150)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- Page Setup ---
st.set_page_config(
    page_title="Intel Scene Classifier | Viva Demo",
    page_icon="🏞️",
    layout="wide" # Use the full width of the page
)

# --- Model Loading ---
@st.cache_resource
def load_trained_model():
    """
    Loads the pre-trained Keras model from disk.
    Uses st.cache_resource to load only once, speeding up the app.
    """
    try:
        # Load the model saved from your Lab 4 notebook
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        # Display a user-friendly error if the model file is missing
        st.error(f"Error loading model: {e}")
        st.error(f"FATAL: Make sure the file '{MODEL_PATH}' is in the same directory as app.py")
        st.stop() # Stop the app if the model can't be loaded

model = load_trained_model()

# --- Image Preprocessing ---
def preprocess_image(image_pil):
    """
    Converts a PIL image into the 4D Tensor format the model expects.
    1. Resizes to (150, 150)
    2. Converts to NumPy array
    3. Rescales pixel values (0-1)
    4. Handles grayscale/Alpha channels
    5. Adds a batch dimension
    """
    img = image_pil.resize(IMG_SIZE)
    img_array = np.array(img)
    
    # Handle edge cases for image channels
    if img_array.ndim == 2: # Grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4: # RGBA (with Alpha channel)
        img_array = img_array[:, :, :3]
        
    img_array = img_array.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0) # [1, 150, 150, 3]
    return img_batch

# --- Sidebar ---
with st.sidebar:
    st.title("🎓 About This Project")
    st.write("**Course:** Neural Networks & Deep Learning")
    st.write("**Lab 4:** CNN for Image Classification")
    st.divider()
    
    st.subheader("Objective")
    st.write("To build, train, and deploy a Convolutional Neural Network (CNN) to classify natural scenes from the Intel Image Classification dataset.")
    
    st.subheader("Model: CNN")
    st.write("The model is a `Sequential` CNN built with `tensorflow.keras`.")
    st.info("The model was trained on ~14,000 images and achieved ~88% accuracy on the test set.")

    st.subheader("Classes (6)")
    st.write(f"{', '.join(CLASS_NAMES)}")

# --- Main Page UI ---
st.title("🏞️ Intel Scene Classifier")
st.write("Upload an image to see the model predict its category in real-time.")

uploaded_file = st.file_uploader(
    "Choose a scene (jpg, jpeg, png)...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload an image to get started.")
else:
    # --- Prediction Logic ---
    image = Image.open(uploaded_file)
    
    with st.spinner("Analyzing image..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        # Get the top prediction
        pred_index = np.argmax(prediction)
        pred_class = CLASS_NAMES[pred_index]
        pred_confidence = np.max(prediction)
        
        # Create the DataFrame for plotting
        prob_df = pd.DataFrame(
            prediction[0],
            index=CLASS_NAMES,
            columns=["Probability"]
        )

    # --- Display Results ---
    st.header(f"Prediction: {pred_class}")
    st.markdown(f"The model is **{pred_confidence * 100:.2f}%** confident.")
    
    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Your Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("Prediction Probabilities")
        st.bar_chart(prob_df)

    # --- Viva Explanation Section ---
    st.divider()
    with st.expander("Click here for Model & Viva Details"):
        st.subheader("Model Architecture")
        st.write("""
        This model is a classic CNN, which is a feedforward network specialized for vision. 
        It learns features hierarchically:
        * **Low-Level Features:** The first layers learn simple edges and colors.
        * **Mid-Level Features:** Deeper layers combine edges to learn textures and patterns.
        * **High-Level Features:** The final layers combine patterns to recognize parts of objects (e.g., "a roof", "a tree trunk").
        
        The final `Dense` layers act as a classifier on these learned features.
        """)
        
        st.code("""
MODEL SUMMARY (from Lab 4):
__________________________________________________
Layer (type)           Output Shape          Param #   
==================================================
conv2d                 (None, 150, 150, 32)  896       
batch_normalization    (None, 150, 150, 32)  128       
max_pooling2d          (None, 75, 75, 32)    0         
__________________________________________________
conv2d_1               (None, 75, 75, 64)    18496     
batch_normalization_1  (None, 75, 75, 64)    256       
max_pooling2d_1        (None, 37, 37, 64)    0         
__________________________________________________
conv2d_2               (None, 37, 37, 128)   73856     
batch_normalization_2  (None, 37, 37, 128)   512       
max_pooling2d_2        (None, 18, 18, 128)   0         
__________________________________________________
conv2d_3               (None, 18, 18, 128)   147584    
batch_normalization_3  (None, 18, 18, 128)   512       
max_pooling2d_3        (None, 9, 9, 128)     0         
__________________________________________________
flatten                (None, 10368)         0         
__________________________________________________
dense                  (None, 512)           5308928   
batch_normalization_4  (None, 512)           2048      
dropout                (None, 512)           0         
__________________________________________________
dense_1 (Output)       (None, 6)             3078      
==================================================
Total params: 5,556,294
        """, language="bash")
        
        st.subheader("What is Backpropagation doing?")
        st.write("""
        When this model was trained, it used **Backpropagation**:
        1.  **Forward Pass:** An image goes in, a prediction comes out.
        2.  **Loss Calculation:** The `categorical_crossentropy` function compares the (wrong) prediction to the (right) label. This gives an "error" number.
        3.  **Backward Pass:** The algorithm calculates the *gradient* (derivative) of this error with respect to every single weight in the model. This tells us "how much" each weight contributed to the error.
        4.  **Weight Update:** The `Adam` optimizer updates all weights in the direction that *minimizes* the error.
        
        This app is the final result of that process: a "frozen" model where the weights are optimized to make correct predictions.
        """)