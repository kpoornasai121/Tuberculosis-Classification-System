
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuration
MODEL_PATH = 'D:/1kpsD9/1Project/MAJOR/majorrr/tb_classifier_model.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Memory optimization settings
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Load the trained model with memory optimization
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    
    try:
        # Reduce memory usage during loading
        with tf.device('/cpu:0'):  # Force CPU loading
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False  # Don't compile during loading
            )
            
            # Recompile with smaller batch size
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
        # st.success("Model loaded successfully with memory optimization")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def predict_tb_image(img):
    """Predict TB from a PIL image with memory optimization."""
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
        return None, None
    
    try:
        # Preprocess with minimal memory usage
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img, dtype=np.float32) / 255.0  # Use float32 instead of float64
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict with memory constraints
        with tf.device('/cpu:0'):
            prediction = model.predict(img_array, verbose=0)[0][0]
            
        class_label = 'Tuberculosis' if prediction > 0.5 else 'Normal'
        confidence = max(prediction, 1 - prediction)  # More efficient calculation
        return class_label, round(float(confidence), 4)  # Convert to native Python float
    except Exception as e:
        st.error(f"Error processing image: {str(e)[:200]}")  # Truncate long error messages
        return None, None

# Streamlit UI
st.title("Tuberculosis Classification from Chest X-rays")
# st.markdown("""
#     Upload a chest X-ray image to classify it as Normal or Tuberculosis.\n
#     Note: This is for research purposes only, not for clinical diagnosis.
# """)

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')  # Ensure RGB format
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Analyzing X-ray..."):
            class_label, confidence = predict_tb_image(image)
            
            if class_label:
                st.subheader("Diagnosis Result")
                if class_label == 'Tuberculosis':
                    st.error(f"⚠️ Tuberculosis detected (confidence: {confidence:.1%})")
                else:
                    st.success(f"✅ Normal (confidence: {confidence:.1%})")
                
                # Add disclaimer
                st.warning("""
                    **Important:** This AI tool is for research purposes only.\n 
                    Always consult a qualified radiologist for medical diagnosis.
                """)
    except Exception as e:
        st.error(f"Error processing your image: {str(e)[:200]}")

# Add footer
st.markdown("---")
st.caption("TB Detection AI is running...")
