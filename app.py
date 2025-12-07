import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import trafilatura
from collections import Counter
import re

st.set_page_config(page_title="News Analyzer", layout="wide")


st.sidebar.header("API Keys")

gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")


st.title("Advanced News Analyzer")
st.write("Summarization • Sentiment • Political Bias • Keyword Extraction • Charts")
st.write("Please enter your **Gemini API Key** in the sidebar to proceed.")

news_url = st.text_input("Enter an English news article URL:")


def extract_article(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None, "Error fetching the article content."
    
    metadata = trafilatura.extract_metadata(downloaded)
    article_text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not article_text:
        return None, "Error extracting the article text."
    
    if metadata and isinstance(metadata, dict):
        title = metadata.get("title", "No Title Found")
    else:
        title = "No Title Found"
    
    return title, article_text


def analyze_with_gemini(prompt):
    if not gemini_api_key:
        st.error("Please enter your Gemini API key.")
        return None

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    response = model.generate_content(prompt)
    return response.text


def extract_keywords(text, top_n=15):
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    common = Counter(tokens).most_common(top_n)
    return common


def plot_sentiment(sentiment_label):
    labels = ["Positive", "Neutral", "Negative"]
    values = [0, 0, 0]

    if sentiment_label.lower().startswith("pos"):
        values[0] = 1
    elif sentiment_label.lower().startswith("neu"):
        values[1] = 1
    else:
        values[2] = 1

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["green", "blue", "red"])
    ax.set_title("Sentiment Classification")
    return fig


def plot_keywords(keyword_list):
    words = [w for w, c in keyword_list]
    counts = [c for w, c in keyword_list]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(words, counts, color="grey")
    ax.set_title("Keyword Frequency")
    ax.invert_yaxis()
    return fig



if st.button("Analyze News"):
    if not news_url:
        st.warning("Please enter a news article URL.")
    else:
        with st.spinner("Analyzing..."):
            title, article_text = extract_article(news_url)

            if not article_text or article_text.startswith("Error"):
                st.error(article_text)
                st.stop()

            st.subheader("Article Title")
            st.write(title)

            st.subheader("Article Text Preview")
            st.write(article_text[:1500] + "...")

            
            prompt = f"""
You are an expert political and media analyst.

Analyze the following NEWS ARTICLE:

{article_text}

TASKS:
1. Provide a concise summary (5-7 sentences).
2. Provide sentiment analysis with:
   - Overall sentiment (Positive, Neutral, or Negative) You do not have to show the words Overall Sentiment.
   - Evidence from the article. You don't have to display the words Evidence from the article.
3. Detect political bias:
   - Classify as Left-leaning, Right-leaning, Center, or Not political.
   - Bold the bias classification.
   - Start a new line in which you explain the political indicators.
4. Provide 5-7 key takeaways.

Only respond in clean sections formatted as below and bold the section titles.:
SUMMARY:
SENTIMENT:
BIAS:
KEY TAKEAWAYS:
"""

            analysis_output = analyze_with_gemini(prompt)

            st.subheader("Summary")
            st.write(analysis_output)

            sentiment_match = re.search(
                r"(Positive|Negative|Neutral)", analysis_output, re.IGNORECASE
            )
            sentiment_label = sentiment_match.group(1) if sentiment_match else "Neutral"

            st.subheader("Sentiment Chart")
            st.pyplot(plot_sentiment(sentiment_label))

            st.subheader("Keyword")
            keywords = extract_keywords(article_text)
            df_kw = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])
            st.table(df_kw)

            st.subheader("Keyword Frequency Chart")
            st.pyplot(plot_keywords(keywords))
