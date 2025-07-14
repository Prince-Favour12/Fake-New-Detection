from transformers import pipeline

# Initialize summarizer once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130):
    if len(text) < 50:
        return None
        
    try:
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )
        return summary[0]['summary_text']
    except:
        return None