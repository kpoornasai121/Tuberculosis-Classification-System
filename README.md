# Tuberculosis Classification System

This project implements a web application for classifying chest X-ray images as either **Normal** or **Tuberculosis** using a deep learning model. The application is built using **Streamlit** for the web interface and **TensorFlow** for loading and running the pre-trained model. The system is designed for research purposes only and is not intended for clinical diagnosis.

## Features
- Upload chest X-ray images in PNG, JPG, or JPEG format.
- Classify images as **Normal** or **Tuberculosis** with confidence scores.
- Memory-optimized model loading and prediction for efficient performance.
- User-friendly interface with a disclaimer emphasizing research use only.

## Dataset
The pre-trained model was developed using the **Tuberculosis (TB) Chest X-ray Dataset** available on Kaggle:  
[Datasets](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

## Prerequisites
To run this application, ensure you have the following installed:
- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`):
  - `streamlit`
  - `tensorflow`
  - `numpy`
  - `pillow`
- Node.js and npm (for running the app with LocalTunnel, optional)
- A pre-trained TensorFlow model (`tb_classifier_model.keras`) stored at the specified path.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kpoornasai121/Tuberculosis-Classification-System
   ```

2. **Install Dependencies**:
   Install the packages manually:
   ```bash
   pip install streamlit tensorflow numpy pillow
   ```

3. **Install LocalTunnel (Optional)**:
   For exposing the Streamlit app to the internet:
   ```bash
   npm install -g localtunnel
   ```

4. **Prepare the Model**:
   Ensure the pre-trained model file (`tb_classifier_model.keras`) is placed in the specified directory. Update the `MODEL_PATH` in `app.py` if your model is stored elsewhere.

## Usage
1. **Run the Streamlit App**:
   Start the Streamlit server:
   ```bash
   streamlit run app.py --server.port 8500
   ```
   The app will be accessible at `http://localhost:8500`.

2. **Expose the App (Optional)**:
   To make the app accessible online, use LocalTunnel:
   ```bash
   lt --port 8500
   ```
   Follow the provided URL to access the app remotely.

3. **Interact with the App**:
   - Open the app in a browser.
   - Upload a chest X-ray image (PNG, JPG, or JPEG).
   - View the classification result ("Normal" or "Tuberculosis") along with the confidence score.
   - Note the disclaimer that the tool is for research purposes only.

## File Structure
- `app.py`: Main Streamlit application script for the web interface and model inference.
- `webapp.ipynb`: Jupyter notebook containing the model loading, prediction function, and initial app development code.
- `tb_classifier_model.keras`: Pre-trained TensorFlow model (not included in the repository; must be provided by the user).
- `requirements.txt`: List of required Python dependencies.
- `README.md`: This file.

## Model Details
- **Input**: Chest X-ray images resized to 224x224 pixels, normalized to [0, 1].
- **Output**: Binary classification ("Normal" or "Tuberculosis") with a confidence score.
- **Model**: Pre-trained TensorFlow model (`tb_classifier_model.keras`) using binary cross-entropy loss and Adam optimizer.
- **Memory Optimization**: 
  - Uses `float32` for reduced memory usage.
  - Forces CPU loading with TensorFlow threading settings (`inter_op_parallelism_threads=2`, `intra_op_parallelism_threads=2`).
  - Employs Streamlit's `@st.cache_resource` for efficient model loading.

## Disclaimer
This tool is intended for **research purposes only** and should not be used for clinical diagnosis. Always consult a qualified radiologist for medical evaluations.

## Contact
For any issues or questions, please open an issue on the repository or contact the maintainers.
