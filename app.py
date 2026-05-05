import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and preprocessing tools
spam_model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
emotion_model = load_model("emotion_model.h5")
tokenizer = joblib.load("tokenizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# Page configuration
st.set_page_config(
    page_title="Smart Email Spam & Emotion Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown(
    """
    <style>
    body {
        background-color: #F9F9F9;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        background-color: #F9D9D9;
        color: #8B5E4E;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }
    .analyze-btn button {
        background-color: #E6B7A9 !important;
        color: #333333 !important;
        font-weight: bold;
        border-radius: 6px;
    }
    .result-box {
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 16px;
    }
    .spam {
        background-color: #F2C9B6;
        color: #660000;
    }
    .not-spam {
        background-color: #F2C9B6;
        color: #006600;
    }
    .emotion {
        background-color: #F2C9B6;
        color: #003366;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar - About Section
st.sidebar.title("📖 About")
st.sidebar.info(
    """
    **Smart Email Spam & Emotion Classifier**  
    This dashboard helps you analyze email content by:
    - Detecting whether an email is **Spam** or **Not Spam**  
    - Identifying the **Emotion** in non-spam emails  
    Built with **Machine Learning & Deep Learning** models for professional use.
    """
)

# Main Title
st.markdown('<div class="main-title">📧 Smart Email Spam & Emotion Classifier</div>', unsafe_allow_html=True)

# Email Input
email_text = st.text_area("Enter your email content:", height=200)

# Analyze Button
if st.button("Analyze", key="analyze-btn"):
    if email_text.strip():
        # Step 1: Spam detection
        clean_email = email_text.lower()
        email_vec = vectorizer.transform([clean_email])
        spam_pred = spam_model.predict(email_vec)[0]

        if spam_pred == 1:
            st.markdown('<div class="result-box spam">🚨 This email is SPAM.</div>', unsafe_allow_html=True)
        else:
            # Step 2: Emotion detection
            seq = tokenizer.texts_to_sequences([clean_email])
            pad = pad_sequences(seq, maxlen=100)
            emotion_pred = emotion_model.predict(pad)
            emotion_label = encoder.inverse_transform([np.argmax(emotion_pred)])[0]
            confidence = np.max(emotion_pred) * 100

            st.markdown('<div class="result-box not-spam">✅ This email is NOT SPAM.</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-box emotion">😊 Emotion detected: <b>{emotion_label}</b> '
                f'(Confidence: {confidence:.2f}%)</div>',
                unsafe_allow_html=True
            )
