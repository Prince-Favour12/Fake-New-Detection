import streamlit as st
import pandas as pd
from datetime import datetime
from detector.fake_news_detector import predict_fake_news
from detector.summarizer import summarize_text
from detector.analyzer import analyze_content
from detector.feedback import save_feedback
from chatbot import NewsChatbot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure application
st.set_page_config(
    page_title="News Insight Analyzer",
    page_icon="📰",
    layout="wide"
)

class NewsAnalyzerApp:
    def __init__(self):
        self.initialize_session_state()
        self.setup_ui()

    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'news_history' not in st.session_state:
            st.session_state.news_history = pd.DataFrame(columns=[
                'title', 'text', 'prediction', 'confidence',
                'genre', 'emotion', 'timestamp'
            ])
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = None

    def setup_ui(self):
        """Set up the main user interface"""
        st.title("📰 News Insight Analyzer with AI Assistant")
        
        # Create main tabs
        self.analysis_tab, self.chat_tab = st.tabs([
            "🔍 Analyze Articles", 
            "💬 Discuss Articles"
        ])
        
        with self.analysis_tab:
            self.render_analysis_interface()
            
        with self.chat_tab:
            self.render_chat_interface()

    def render_analysis_interface(self):
        """Render the article analysis interface"""
        self.analyze_new_article()
        self.display_article_history()

    def analyze_new_article(self):
        """Handle article analysis form and processing"""
        with st.form("news_form", border=True):
            st.subheader("Analyze New Article")
            title = st.text_input(
                "Article Title*", 
                placeholder="Enter news headline",
                help="The title of the news article to analyze"
            )
            text = st.text_area(
                "Article Content*", 
                height=250,
                placeholder="Paste the full article content here...",
                help="The actual content of the news article"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                author = st.text_input("Author", placeholder="Optional")
            with col2:
                source = st.text_input("Source", placeholder="Optional")
            
            submitted = st.form_submit_button(
                "Analyze Article", 
                use_container_width=True
            )
            st.caption("*Required fields")

        if submitted:
            self.process_article_submission(title, text, author, source)

    def process_article_submission(self, title, text, author, source):
        """Process a new article submission"""
        if not title or not text:
            st.warning("Please provide both title and content for analysis")
            return

        with st.status("Analyzing article content...", expanded=True) as status:
            try:
                # Perform all analyses
                st.write("🔍 Evaluating authenticity...")
                predictions = predict_fake_news(text)
                is_fake = predictions['label'] == 'FAKE'
                conf_score = predictions.get('score', 0.0)
                prediction_label = "Fake" if is_fake else "Real"
                
                st.write("📊 Analyzing content characteristics...")
                analysis = analyze_content(text)
                summary = summarize_text(text)
                
                # Store results
                self.store_analysis_results(
                    title, text, author, source,
                    prediction_label, conf_score, analysis
                )
                
                status.update(label="Analysis complete!", state="complete")
                self.display_analysis_results(
                    title, text, prediction_label, 
                    conf_score, analysis, summary
                )
                
            except Exception as e:
                logger.error(f"Analysis failed: {str(e)}")
                status.update(label="Analysis failed!", state="error")
                st.error("An error occurred during analysis. Please try again.")

    def store_analysis_results(self, title, text, author, source, prediction, confidence, analysis):
        """Store analysis results in session state"""
        new_entry = {
            'title': title,
            'text': text,
            'prediction': prediction,
            'confidence': f"{confidence:.0%}",
            'genre': analysis.get('genre', 'N/A'),
            'emotion': analysis.get('emotion', 'N/A'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        
        st.session_state.news_history = pd.concat([
            st.session_state.news_history,
            pd.DataFrame([new_entry])
        ], ignore_index=True)
        
        # Reset chatbot to ensure it uses updated data
        st.session_state.chatbot = None

    def display_analysis_results(self, title, text, prediction, confidence, analysis, summary):
        """Display the results of article analysis"""
        st.subheader("Analysis Results", divider="rainbow")
        
        cols = st.columns(4)
        cols[0].metric(
            "Authenticity", 
            prediction,
            "⚠️ Verify sources" if prediction == "Fake" else "✅ Likely credible"
        )
        cols[1].metric("Confidence", confidence)
        cols[2].metric("Content Length", f"{len(text.split())} words")
        cols[3].metric("Reading Level", analysis.get('reading_level', 'N/A'))
        
        st.subheader("Content Insights")
        with st.expander("Show detailed analysis"):
            tab_a, tab_b, tab_c = st.tabs(["📊 Metrics", "📝 Summary", "🧐 Feedback"])
            
            with tab_a:
                st.json({
                    "Genre": analysis.get('genre'),
                    "Primary Emotion": analysis.get('emotion'),
                    "Tone": analysis.get('tone'),
                    "Subjectivity": analysis.get('subjectivity'),
                    "Key Topics": analysis.get('topics', [])
                })
            
            with tab_b:
                if summary:
                    st.info(summary)
                else:
                    st.warning("No summary generated")
            
            with tab_c:
                with st.form("feedback_form"):
                    feedback = st.text_area(
                        "Your evaluation",
                        placeholder="Do you agree with this analysis?"
                    )
                    accuracy = st.select_slider(
                        "Accuracy rating",
                        options=[
                            "Very inaccurate", "Somewhat inaccurate",
                            "Neutral", "Somewhat accurate",
                            "Very accurate"
                        ]
                    )
                    
                    if st.form_submit_button("Submit Evaluation"):
                        save_feedback(title, feedback, accuracy)
                        st.toast("Feedback saved successfully!")

    def display_article_history(self):
        """Show history of analyzed articles"""
        st.subheader("📚 Your Analyzed Articles", divider="gray")
        
        if st.session_state.news_history.empty:
            st.info("No articles analyzed yet. Get started by analyzing your first article above.")
            return
        
        search_query = st.text_input("Search your articles", key="history_search")
        
        filtered_history = st.session_state.news_history
        if search_query:
            filtered_history = filtered_history[
                filtered_history['title'].str.contains(search_query, case=False) |
                filtered_history['text'].str.contains(search_query, case=False)
            ]
        
        st.dataframe(
            filtered_history.sort_values('timestamp', ascending=False),
            column_config={
                "timestamp": "Analysis Date",
                "prediction": st.column_config.TextColumn("Veracity"),
                "confidence": "Confidence",
                "genre": "Category"
            },
            hide_index=True,
            use_container_width=True
        )

    def render_chat_interface(self):
        """Render the chat interface"""
        st.subheader("💬 Discuss Analyzed Articles", divider="blue")
        
        if st.session_state.news_history.empty:
            self.render_empty_chat_state()
            return
        
        self.initialize_chatbot()
        self.display_chat_history()
        self.handle_chat_input()
        self.display_suggested_questions()

    def render_empty_chat_state(self):
        """Render UI when no articles are available"""
        st.warning("No articles available for discussion yet.")
        st.info("Analyze some articles first to enable this feature.")
        if st.button("Analyze Articles Now"):
            st.switch_page("app.py")

    def initialize_chatbot(self):
        """Initialize or reuse the chatbot instance"""
        if st.session_state.chatbot is None:
            try:
                with st.spinner("Loading AI assistant..."):
                    st.session_state.chatbot = NewsChatbot(st.session_state.news_history)
            except Exception as e:
                st.error(f"Failed to initialize chatbot: {str(e)}")
                logger.error(f"Chatbot initialization failed: {str(e)}")

    def display_chat_history(self):
        """Display the chat message history"""
        for msg in st.session_state.chat_messages:
            avatar = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                
                if msg.get("reference"):
                    with st.expander("📄 Referenced Article"):
                        st.write(f"**Title:** {msg['reference']['title']}")
                        st.write(f"**Verdict:** {msg['reference']['prediction']} ({msg['reference']['confidence']})")
                        st.write(f"**Excerpt:** {msg['reference']['text']}")

    def handle_chat_input(self):
        """Handle user chat input and generate responses"""
        if prompt := st.chat_input("Ask about any analyzed article..."):
            self.process_user_message(prompt)

    def process_user_message(self, prompt):
        """Process a user message and generate response"""
        # Add user message to history
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate and display response
        with st.spinner("Analyzing your question..."):
            try:
                if st.session_state.chatbot:
                    response, reference = st.session_state.chatbot.answer_question(prompt)
                    self.add_assistant_response(response, reference)
                else:
                    self.add_assistant_response(
                        "Chatbot not initialized. Please try again.",
                        None
                    )
            except Exception as e:
                logger.error(f"Chat processing failed: {str(e)}")
                self.add_assistant_response(
                    "Sorry, I encountered an error processing your request.",
                    None
                )
        
        st.rerun()

    def add_assistant_response(self, response, reference):
        """Add an assistant response to the chat history"""
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response,
            "reference": reference
        })

    def display_suggested_questions(self):
        """Show suggested questions to start conversations"""
        st.caption("Try asking:")
        cols = st.columns(3)
        questions = [
            "What was the last article about?",
            "Find articles about current events",
            "Summarize the key points from recent articles"
        ]
        for col, question in zip(cols, questions):
            if col.button(question, use_container_width=True):
                self.process_user_message(question)

if __name__ == "__main__":
    app = NewsAnalyzerApp()