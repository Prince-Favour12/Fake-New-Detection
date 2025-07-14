import joblib
import numpy as np
from gensim.models import Word2Vec
from utils import tokenize, document_to_vector

# Load models
w2v_model = Word2Vec.load("models/word2vec.model")
clf = joblib.load("models/word2vec_clf.pkl")

def predict_fake_news(text):
    """Predict using Word2Vec embeddings"""
    tokens = tokenize(text)
    vector = document_to_vector(tokens, w2v_model).reshape(1, -1)
    proba = clf.predict_proba(vector)[0]
    is_fake = clf.predict(vector)[0] == 0
    confidence = np.max(proba)
    return is_fake, float(confidence)