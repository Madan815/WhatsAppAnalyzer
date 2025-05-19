import streamlit as st
from whatstk import df_from_whatsapp
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
import openai
import tempfile

# ---- Modern UI: Custom CSS ----
st.markdown("""
    <style>
        body {
            background-color: #f8fafc;
        }
        .main {
            background-color: #ffffff;
        }
        .stApp {
            background-color: #f8fafc;
        }
        .blue-header {
            color: #0a2540;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: #1e293b;
            font-size: 1.2rem;
            margin-bottom: 1.5rem;
        }
        .feedback-box {
            background-color: #e0f2fe;
            border-radius: 10px;
            padding: 1rem;
            color: #0a2540;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        .summary-box {
            background-color: #dcfce7;
            border-radius: 10px;
            padding: 1rem;
            color: #14532d;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        .footer {
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 2rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Your existing functions ----

nlp = spacy.load("en_core_web_sm")

interest_keywords = ["interest", "want", "apply", "yes", "sure", "how", "detail", "send", "okay", "please", "share", "process", "offer"]
disinterest_keywords = ["not interest", "no", "don't want", "not now", "maybe later", "stop", "never", "don't call", "not require", "not need"]

def detect_interest_nlp(text):
    doc = nlp(str(text).lower())
    lemmas = " ".join([token.lemma_ for token in doc])
    for word in interest_keywords:
        if word in lemmas:
            return "Interested"
    for word in disinterest_keywords:
        if word in lemmas:
            return "Not Interested"
    return "Neutral"

def generate_summary(df):
    interest_counts = df['interest_level_nlp'].value_counts().to_dict()
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    interested_count = interest_counts.get('Interested', 0)
    not_interested_count = interest_counts.get('Not Interested', 0)
    neutral_count = interest_counts.get('Neutral', 0)

    if interested_count > max(not_interested_count, neutral_count):
        interest_summary = "The customer is interested to buy."
    elif not_interested_count > max(interested_count, neutral_count):
        interest_summary = "The customer is not interested to buy."
    else:
        interest_summary = "The customer's interest is neutral or unclear."

    positive_sentiment = sentiment_counts.get('Positive', 0)
    negative_sentiment = sentiment_counts.get('Negative', 0)
    neutral_sentiment = sentiment_counts.get('Neutral', 0)

    if positive_sentiment > max(negative_sentiment, neutral_sentiment):
        sentiment_summary = "Overall sentiment is positive."
    elif negative_sentiment > max(positive_sentiment, neutral_sentiment):
        sentiment_summary = "Overall sentiment is negative."
    else:
        sentiment_summary = "Overall sentiment is neutral."

    summary = f"{interest_summary} {sentiment_summary}"
    return summary

def analyze_whatsapp_chat(input_path):
    df = df_from_whatsapp(input_path)
    sentiment_analyzer = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    def get_sentiment(text):
        result = sentiment_analyzer(str(text))[0]
        return result['label'], result['score']

    df[['sentiment', 'sentiment_score']] = df['message'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )
    df['interest_level_nlp'] = df['message'].apply(detect_interest_nlp)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['interest_level_nlp']
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    df['predicted_interest'] = clf.predict(vectorizer.transform(df['message']))
    return df

def get_seller_feedback(chat_df, seller_name, api_key_mistral):
    seller_msgs = chat_df[chat_df['username'] == seller_name]['message'].tolist()
    if not seller_msgs:
        return "No seller messages found in the chat."
    chat_text = "\n".join(seller_msgs[-15:])

    prompt = (
        "You are a sales coach. Analyze the following WhatsApp sales messages sent by a seller to a customer. "
        "Give actionable, constructive feedback on how the seller can improve their sales pitch. "
        "Point out any mistakes, missed opportunities, or negative language. "
        "Be specific and helpful.\n\n"
        f"Seller's messages:\n{chat_text}\n\n"
        "Suggestions:"
    )

    client = openai.OpenAI(
        api_key=API_KEY_MISTRAL,
        base_url="https://api.mistral.ai/v1"
    )

    response = client.chat.completions.create(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# ---- Streamlit App ----

# Hardcoded API key (for demo only; use env variable or secrets in production)
API_KEY_MISTRAL = "aKFEMuDwJOvtphHDDOrh2qbfRP7jEA1L"

st.markdown('<div class="blue-header">WhatsApp Sales Chat Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your WhatsApp exported chat and get instant sales analysis and actionable feedback!</div>', unsafe_allow_html=True)

seller_name = st.text_input("Enter the seller's name (as it appears in chat):")
uploaded_file = st.file_uploader("Upload WhatsApp exported chat (.txt file)", type=["txt"])

if uploaded_file and seller_name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    with st.spinner("Analyzing chat..."):
        df_result = analyze_whatsapp_chat(tmp_file_path)
        summary_output = generate_summary(df_result)
        feedback = get_seller_feedback(df_result, seller_name, API_KEY_MISTRAL)

    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.subheader("Summary for Seller")
    st.write(summary_output)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
    st.subheader("Sales Pitch Feedback for Seller")
    st.write(feedback)
    st.markdown('</div>', unsafe_allow_html=True)
elif uploaded_file:
    st.warning("Please enter the seller's name.")

st.markdown('<div class="footer">Made with ❤️ using Streamlit, Mistral AI, and Hugging Face Transformers</div>', unsafe_allow_html=True)
