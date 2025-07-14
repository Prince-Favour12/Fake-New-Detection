import json
from datetime import datetime
import os
from transformers import pipeline

# Initialize sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis")

def save_feedback(article_title, user_feedback, prediction_correct):
    if not user_feedback:
        return {}
    
    # Analyze sentiment
    sentiment = sentiment_analyzer(user_feedback[:512])[0]
    
    # Prepare data
    feedback_data = {
        "article": article_title,
        "feedback": user_feedback,
        "correct": prediction_correct,
        "sentiment": sentiment['label'],
        "score": sentiment['score'],
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to file
    os.makedirs("data/feedback", exist_ok=True)
    filename = f"data/feedback/feedback_{datetime.now().timestamp()}.json"
    with open(filename, "w") as f:
        json.dump(feedback_data, f)
    
    # Generate response
    if sentiment['label'] == 'POSITIVE':
        response = "Thank you for your positive feedback!"
    elif prediction_correct:
        response = "Thanks for confirming our prediction!"
    else:
        response = "We'll use your feedback to improve. Thank you!"
    
    return {
        "sentiment": sentiment['label'],
        "response": response
    }