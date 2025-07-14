from transformers import pipeline

classifier = pipeline(
    "text-classification", 
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
    )


label_map = {'LABEL_0': 'REAL', 'LABEL_1': 'FAKE'}

def predict_fake_news(text):
    results = classifier(text)[0]
    results['label'] = label_map.get(results['label'], results['label'])
    return results

if __name__ == "__main__":
    sample_text = "This is a sample text to test the fake news detection."
    prediction = predict_fake_news(sample_text)
    print(f"Text: {sample_text}\nPrediction: {prediction}")