# import streamlit as st
# import pandas as pd
# from detector.fake_news_detector import predict_fake_news
# from detector.summarizer import summarize_text
# from detector.analyzer import analyze_content
# from detector.feedback import save_feedback

# st.title("📰 Multi-Column News Analyzer")

# # Input form with multiple fields
# with st.form("news_form"):
#     title = st.text_input("News Title")
#     text = st.text_area("Main Content")
#     author = st.text_input("Author (Optional)")
#     source = st.text_input("Source (Optional)")
#     submitted = st.form_submit_button("Analyze")

# if submitted:
#     # Create dataframe from inputs
#     input_data = pd.DataFrame([{
#         'title': title,
#         'text': text,
#         'author': author,
#         'source': source
#     }])
    
#     # Analyze using multiple columns
#     predictions_dict= predict_fake_news(text)
#     is_fake = predictions_dict['label'] == 'FAKE'
#     conf_score = predictions_dict.get('Confidence', 0.0)
    
#     # Display results
#     st.subheader("Analysis Results")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.metric("Prediction", "Fake" if is_fake else "Real")
#     with col2:
#         st.metric("Confidence", f"{conf_score:.0%}")
    
#     # Content analysis
#     if text:
#         analysis = analyze_content(text)
#         summary = summarize_text(text)
        
#         st.subheader("Content Insights")
#         st.write(f"**Genre**: {analysis.get('genre', 'N/A')}")
#         st.write(f"**Emotion**: {analysis.get('emotion', 'N/A')}")
        
#         if summary:
#             st.subheader("Summary")
#             st.info(summary)

#     # Feedback system
#     with st.expander("Provide Feedback"):
#         with st.form("feedback_form"):
#             feedback = st.text_area("Your feedback")
#             correct = st.radio("Was this prediction accurate?", ["Yes", "No"])
#             if st.form_submit_button("Submit"):
#                 result = save_feedback(
#                     title,
#                     feedback,
#                     correct == "Yes"
#                 )
#                 if result:
#                     st.success("Thank you for your feedback!")