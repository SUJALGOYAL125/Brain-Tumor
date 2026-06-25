import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('brain_tumor_model.h5')

model = load_model()

# ============ CLASS NAMES ============
CLASS_NAMES = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor',
    2: 'No Tumor',
    3: 'Pituitary Tumor'
}

# ============ UI ============
st.title("🧠 Brain Tumor Detection")
st.markdown("### AI Powered MRI Brain Scan Analyzer")
st.markdown("---")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app uses Transfer Learning (VGG16)
to detect brain tumors from MRI scans.

**Tumor Types:**
- 🔴 Glioma Tumor
- 🟠 Meningioma Tumor
- 🟡 Pituitary Tumor
- 🟢 No Tumor
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** VGG16 Transfer Learning")
st.sidebar.markdown("**Accuracy:** 95%+")
st.sidebar.markdown("**Classes:** 4")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose MRI Image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

with col2:
    st.markdown("### 📊 Detection Result")

    if uploaded_file:
        if st.button("🔍 Detect Tumor", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing MRI Scan..."):
                # Preprocess image
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class] * 100

                # Show result
                st.markdown("---")
                if predicted_class == 2:  # No Tumor
                    st.success(f"✅ {CLASS_NAMES[predicted_class]}")
                    st.balloons()
                else:
                    st.error(f"⚠️ {CLASS_NAMES[predicted_class]}")

                st.metric("Confidence", f"{confidence:.2f}%")

                # Probability bars
                st.markdown("#### All Probabilities")
                for i, class_name in enumerate(CLASS_NAMES.values()):
                    prob = predictions[0][i] * 100
                    st.progress(
                        int(prob),
                        text=f"{class_name}: {prob:.2f}%"
                    )

            st.info("⚕️ AI prediction only. Consult a doctor!")
    else:
        st.markdown("""
        #### How to use:
        1. Upload MRI brain scan image
        2. Click 'Detect Tumor' button
        3. View detection results
        
        #### Detects:
        - 🔴 Glioma Tumor
        - 🟠 Meningioma Tumor
        - 🟡 Pituitary Tumor
        - 🟢 No Tumor
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>Built with ❤️ using VGG16 Transfer Learning</p>",
    unsafe_allow_html=True
)