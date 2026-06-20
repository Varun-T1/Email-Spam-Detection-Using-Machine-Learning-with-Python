import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Load augmented dataset
p = Path('data/spam_aug_8000.csv')
if not p.exists():
    raise FileNotFoundError(f"Augmented dataset not found: {p}")

df = pd.read_csv(p)
# ensure label column normalized
df['label'] = df['label'].str.lower().map({'spam':1, 'ham':0})
X = df['text'].fillna('').astype(str)
y = df['label'].astype(int)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# vectorizer + model
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_t = vectorizer.fit_transform(X_train)
X_test_t = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_t, y_train)

# eval
pred = model.predict(X_test_t)
acc = accuracy_score(y_test, pred)
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test, pred, target_names=['ham','spam']))

# Save model and vectorizer
Path('models').mkdir(exist_ok=True)
joblib.dump(model, 'models/spam_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print('Saved model to models/spam_model.pkl and vectorizer to models/vectorizer.pkl')
