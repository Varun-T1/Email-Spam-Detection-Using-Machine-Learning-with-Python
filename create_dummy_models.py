import os
import joblib
from dummy_models import DummyModel, DummyVectorizer

os.makedirs('models', exist_ok=True)
vec = DummyVectorizer()
model = DummyModel()
joblib.dump(model, os.path.join('models', 'spam_model.pkl'))
joblib.dump(vec, os.path.join('models', 'vectorizer.pkl'))
print('Dummy models written to models/')
