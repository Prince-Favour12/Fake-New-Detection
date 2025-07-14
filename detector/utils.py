import re
import html
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

def clean_text(text):
    """Enhanced cleaning for Word2Vec"""
    if not isinstance(text, str):
        return ""
    
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
    return text.lower().strip()

def tokenize(text):
    """Tokenize text for Word2Vec"""
    return simple_preprocess(clean_text(text))

def document_to_vector(tokens, model):
    """Convert document to average vector"""
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)