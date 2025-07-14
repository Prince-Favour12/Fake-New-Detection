from transformers import pipeline

# Initialize models once
genre_classifier = pipeline("zero-shot-classification")
emotion_detector = pipeline("text-classification", 
                           model="j-hartmann/emotion-english-distilroberta-base")

def analyze_content(text):
    if not text:
        return {}
    
    # Genre detection
    genre_labels = ["technology", "politics", "entertainment", "sports", "business"]
    genre = genre_classifier(text, genre_labels)['labels'][0]
    
    # Emotion detection
    emotion = emotion_detector(text)[0]['label']
    
    return {
        "genre": genre,
        "emotion": emotion
    }