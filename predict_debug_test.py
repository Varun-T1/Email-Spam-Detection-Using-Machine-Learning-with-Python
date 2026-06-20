from utils import load_model
from preprocess import clean_text

model, vectorizer = load_model("models/spam_model.pkl", "models/vectorizer.pkl")

samples = [
    "You have won a FREE ticket! Claim your prize now!",
    "Hi, are we still meeting today?",
    "WIN a free entry, spam offer!!",
    "This is definitely not spam, just a test",
]

print("Model type:", type(model))
print("Vectorizer type:", type(vectorizer))

for s in samples:
    cleaned = clean_text(s)
    transformed = vectorizer.transform([cleaned])
    pred = model.predict(transformed)
    print('\nSample:', s)
    print('Cleaned:', cleaned)
    print('Transformed (type):', type(transformed))
    # try to print content for common types
    try:
        # if sparse / array-like, show shape
        import numpy as np
        if hasattr(transformed, 'shape'):
            print('Transformed shape:', transformed.shape)
    except Exception:
        pass
    print('Raw model.predict output:', pred)
    print('Interpreted:', 'SPAM' if (pred[0] == 1 or pred == 1) else 'HAM')
