# 🧠 Brain Tumor Detection

> AI powered brain tumor detection from MRI scans using Transfer Learning (VGG16)

🌐 **Live Demo:** [Click here to try](your-streamlit-link)
📁 **GitHub:** [Repository](your-github-link)

---

## ✨ Features

- 🧠 Detects 4 types of brain tumors from MRI scans
- 🎯 91.31% test accuracy
- 📊 Probability distribution for all classes
- ⚡ Real time prediction
- 🌐 Deployed on Streamlit Cloud

---

## 🤖 Model Details

| Feature | Details |
|---|---|
| Architecture | VGG16 Transfer Learning |
| Dataset | Brain Tumor MRI Dataset (Kaggle) |
| Training Images | 4480 |
| Testing Images | 1600 |
| Test Accuracy | 91.31% |

---

## 🎯 Per Class Accuracy

| Class | Accuracy |
|---|---|
| Glioma Tumor | 77.5% |
| Meningioma Tumor | 88.0% |
| No Tumor | 99.8% |
| Pituitary Tumor | 100.0% |

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- VGG16 (ImageNet weights)
- NumPy
- Pillow
- gdown

---

## 📁 Project Structure
BrainTumorDetection/

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

└── brain_tumor_training.ipynb

---

## 🚀 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 📊 Dataset

- **Source:** Kaggle — Brain Tumor MRI Dataset
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Total Images:** 7023

---

## 🧠 How It Works
Upload MRI Image

↓

Preprocess (224x224)

↓

VGG16 Feature Extraction

↓

Custom Dense Layers

↓

Tumor Classification

↓

Result + Confidence Score


---

## ⚠️ Disclaimer

This app is for educational purposes only.
Not a substitute for professional medical diagnosis.
Always consult a qualified doctor.

---

