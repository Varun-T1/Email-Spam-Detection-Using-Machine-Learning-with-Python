import sys
from preprocess import clean_text
from utils import load_model

def predict_message(message):
    model, vectorizer = load_model("models/spam_model.pkl", "models/vectorizer.pkl")
    cleaned = clean_text(message)
    text_vector = vectorizer.transform([cleaned])
    prediction = model.predict(text_vector)[0]
    return "SPAM" if prediction == 1 else "HAM"

if __name__ == "__main__":
    msg = sys.argv[1]
    print("Prediction:", predict_message(msg))
