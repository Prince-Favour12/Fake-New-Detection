import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class NewsChatbot:
    def __init__(self, news_data):
        self.news_data = news_data.copy()
        self.qa_model = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"
        )
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        if not self.news_data.empty:
            self.news_data['embedding'] = self.news_data['text'].apply(
                lambda x: self.embedder.encode(str(x)))
        else:
            self.news_data['embedding'] = []

    def find_most_relevant(self, query):
        """Always returns a dataframe row, even if empty"""
        if self.news_data.empty:
            return pd.Series(dtype=object)
            
        query_embed = self.embedder.encode(query)
        self.news_data['similarity'] = self.news_data['embedding'].apply(
            lambda x: cosine_similarity([x], [query_embed])[0][0])
        return self.news_data.nlargest(1, 'similarity').iloc[0]

    def answer_question(self, question):
        """Returns tuple of (response, reference_dict) in all cases"""
        try:
            if self.news_data.empty:
                return "No articles available to analyze. Please add some articles first.", None
                
            article = self.find_most_relevant(question)
            
            if pd.isna(article['title']):  # No relevant article found
                return "I couldn't find a relevant article. Try analyzing more content.", None
                
            context = f"""
            Title: {article['title']}
            Content: {article['text']}
            Analysis Results:
            - Verdict: {article['prediction']} ({article['confidence']})
            - Category: {article['genre']}
            - Emotion: {article['emotion']}
            """
            
            answer = self.qa_model(
                question=question,
                context=context,
                max_answer_len=200
            )
            
            reference = {
                'title': article['title'],
                'prediction': article['prediction'],
                'confidence': article['confidence'],
                'text': article['text'][:500] + "..."
            }
            
            if answer['score'] > 0.4:
                return answer['answer'], reference
            else:
                return f"This article discusses: {article['text'][:300]}...", reference
                
        except Exception as e:
            error_msg = "I encountered an error processing your request. Please try again."
            return error_msg, None