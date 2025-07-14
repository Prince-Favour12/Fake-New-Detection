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
    

if __name__ == '__main__':
    text = "The local government announced the completion of a major bridge project connecting two key districts, improving traffic flow and boosting commerce."
    print(summarize_text(text, 50))