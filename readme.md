# 📰 News Insight Analyzer with AI Assistant

**Detect fake news, understand sentiment & categories, and chat with your articles powered by advanced AI models.**

---

## 📌 Overview

**News Insight Analyzer** is an intelligent web app that helps users:
- **Detect fake or real news** using state-of-the-art NLP models.
- Get a **confidence score or probability** for predictions.
- Analyze **emotion and category** of any article.
- Collect **user feedback** and perform sentiment analysis on it.
- Provide a **dynamic AI assistant** that allows users to *chat with their article* for deeper understanding.
- Store **past articles and results** for reference.

---

## ⚙️ Tools & Libraries Used

- 🤗 **Hugging Face Transformers** – Pre-trained language models for classification & embeddings
- 🧠 **Sentence Transformers** – For semantic search & contextual understanding
- 🐼 **Pandas** – Data handling & feedback storage
- 🌐 **Streamlit** – Interactive web app UI
- 🔥 **PyTorch** – Deep learning backend
- 📊 **Matplotlib** – Visualizing insights & sentiment
- ⚙️ **scikit-learn** – Additional ML utilities & evaluation

---

## 🚀 Features

✅ **Fake News Detection**  
Reads any news article and predicts whether it’s *Real* or *Fake*, along with a confidence score.

✅ **Emotion & Category Classification**  
Uses pre-trained NLP models to analyze the emotion and categorize the article (e.g., Politics, Tech, Health).

✅ **Feedback & Sentiment Analysis**  
Users can leave feedback on predictions. The app analyzes feedback sentiment (Positive, Neutral, Negative) and responds accordingly.

✅ **Article Archive**  
Automatically saves all analyzed articles and results for later review.

✅ **AI Assistant Chat**  
Interact with your article ask questions, get clarifications, and explore insights through an integrated AI-powered chat assistant.

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/news-insight-analyzer.git
   cd news-insight-analyzer
    ```
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**
   ```bash
    pip install -r requirements.txt
    ```
4. **Run the app**
    ```bash
    streamlit run home.py
    ```
