import streamlit as st
import sys
import os

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text
from utils import load_model

@st.cache_resource
def load_spam_model():
    return load_model("models/spam_model.pkl", "models/vectorizer.pkl")

model, vectorizer = load_spam_model()

st.title("📧 Email Spam Detector")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)[0]
    st.write("### Result: **Spam**" if result == 1 else "### Result: **Ham**")
