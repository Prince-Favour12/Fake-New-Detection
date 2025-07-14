import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from utilis import preprocess_dataframe

def load_and_prepare_data(file_path, text_columns=['text', 'title']):
    df = pd.read_csv(file_path)
    df = preprocess_dataframe(df, text_columns)
    
    # Combine text columns
    df['combined_text'] = df[text_columns[0]]
    for col in text_columns[1:]:
        df['combined_text'] += " [SEP] " + df[col]
    
    return df['combined_text'], df['label']

def train_model():
    X, y = load_and_prepare_data("Fake.csv", "True.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_df=0.7,
            ngram_range=(1, 2)),  # Using bigrams
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced')))  # Handle imbalanced data
    ])
    
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    
    joblib.dump(model, "models/fake_news_model.pkl")

if __name__ == "__main__":
    train_model()