import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Streamlit Page Config
st.set_page_config(page_title='Pulmonary Embolism Detection (CNN)', layout='centered')

# Sidebar Navigation
with st.sidebar:
    selected = st.selectbox('Navigation', ['Introduction', 'CNN Based Embolism Prediction'])

# ------------------ INTRO PAGE ------------------
if selected == 'Introduction':
    st.title("🧠 Lung Cancer & Pulmonary Embolism Detection App")

    st.subheader("📌 What is Pulmonary Embolism?")
    st.write("""
        Pulmonary Embolism (PE) is a serious condition that occurs when one or more arteries in the lungs become blocked by a blood clot.
        These clots usually travel from the legs or other parts of the body (deep vein thrombosis).

        🩸 Causes:
        - Blood clots (most common)
        - Long periods of immobility (e.g., long flights or hospital stays)
        - Certain medical conditions (e.g., cancer, heart disease)

        ⚠️ Symptoms:
        - Sudden shortness of breath  
        - Chest pain that worsens with deep breathing  
        - Rapid heart rate  
        - Coughing, sometimes with blood  

        🧠 Early detection through imaging such as CT pulmonary angiography or CT scans can be life-saving.
    """)

    st.subheader("🧪 About This App")
    st.write("""
        This app uses a trained Convolutional Neural Network (CNN) to analyze CT scan images for signs of Pulmonary Embolism and provide
        basic estimation of the affected lung region.

        ✅ Model: Trained CNN using Keras  
        📷 Input: CT Scan Image (JPG/PNG)  
        🔍 Output: Embolism Prediction + Estimated Lung Side (if applicable)  
    """)

# ------------------ PREDICTION PAGE ------------------
elif selected == 'CNN Based Embolism Prediction':
    st.title(" Lung Embolism Detection using CNN and CT-Scan Images")

    @st.cache(allow_output_mutation=True)
    def load_cnn_model():
        model_path = "models/keras_model.h5"
        if not os.path.exists(model_path):
            st.error(" Model file not found at: models/keras_model.h5")
            return None
        return load_model(model_path)

    cnn = load_cnn_model()
    if cnn is None:
        st.stop()

    model_input_shape = cnn.input_shape  # e.g., (None, 150, 150, 1)
    st.write(f"📐 Model expects input shape: `{model_input_shape}`")

    uploaded_file = st.file_uploader(" Upload CT Scan Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption=' Uploaded CT Scan', use_column_width=True)

        img = Image.open(uploaded_file)
        target_size = (model_input_shape[1], model_input_shape[2])

        if model_input_shape[3] == 1:
            img = img.convert("L")  # Grayscale
        else:
            img = img.convert("RGB")  # RGB

        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized) / 255.0

        if model_input_shape[3] == 1:
            img_array = np.expand_dims(img_array, axis=-1)

        img_array = np.expand_dims(img_array, axis=0)

        try:
            preds = cnn.predict(img_array)
            pred_prob = preds[0][0]

            show_lung_side = True  # Flag to conditionally show lung side

            if pred_prob >= 0.9:
                st.success(f"🟢 Likely Normal — Confidence: {pred_prob:.2%}\nEmbolism Intensity: Negligible")
                show_lung_side = False  # Don't show lung side if Embolism is negligible
            elif pred_prob >= 0.7:
                st.info(f"🟡 Low Embolism Intensity — Confidence: {pred_prob:.2%}")
            elif pred_prob >= 0.4:
                st.warning(f"🟠 Moderate Embolism Intensity — Confidence: {(1 - pred_prob):.2%}")
            else:
                st.error(f"🔴 High Embolism Intensity — Confidence: {(1 - pred_prob):.2%}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {repr(e)}")
            st.stop()

        # Only estimate lung side if needed
        if show_lung_side:
            try:
                grayscale = np.array(img_resized.convert("L"))
                left_half = grayscale[:, :grayscale.shape[1] // 2].sum()
                right_half = grayscale[:, grayscale.shape[1] // 2:].sum()

                if abs(left_half - right_half) < 0.1 * (left_half + right_half):
                    lung_location = "Both Lungs"
                elif left_half > right_half:
                    lung_location = "Left Lung"
                else:
                    lung_location = "Right Lung"

                st.write(f"🫁 **Predicted Lung Side**: {lung_location}")

            except Exception as e:
                st.warning(f"⚠️ Lung side estimation failed: {e}")
