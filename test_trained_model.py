from preprocess import clean_text
from utils import load_model

model, vectorizer = load_model('models/spam_model.pkl', 'models/vectorizer.pkl')

test_messages = [
    'You won a free ticket claim now!',
    'Hi there how are you today?',
    'WINNER congratulations free prize entry',
    'Let us meet tomorrow for coffee'
]

print('Testing trained model:')
for msg in test_messages:
    cleaned = clean_text(msg)
    pred = model.predict(vectorizer.transform([cleaned]))[0]
    result = 'SPAM' if pred == 1 else 'HAM'
    print(f'  "{msg}" -> {result}')
