from whatstk import df_from_whatsapp
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
import openai  # <-- Add this import

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Keywords for interest detection (optional)
interest_keywords = ["interest", "want", "apply", "yes", "sure", "how", "detail", "send", "okay", "please", "share", "process", "offer"]
disinterest_keywords = ["not interest", "no", "don't want", "not now", "maybe later", "stop", "never", "don't call", "not require", "not need"]

# Interest detection using NLP keywords (optional)
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

# Summary generator for seller
def generate_summary(df):
    # Count the number of interest labels
    interest_counts = df['interest_level_nlp'].value_counts().to_dict()
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    interested_count = interest_counts.get('Interested', 0)
    not_interested_count = interest_counts.get('Not Interested', 0)
    neutral_count = interest_counts.get('Neutral', 0)

    # Determine overall interest
    if interested_count > max(not_interested_count, neutral_count):
        interest_summary = "The customer is interested to buy."
    elif not_interested_count > max(interested_count, neutral_count):
        interest_summary = "The customer is not interested to buy."
    else:
        interest_summary = "The customer's interest is neutral or unclear."

    # Determine overall sentiment
    positive_sentiment = sentiment_counts.get('Positive', 0)
    negative_sentiment = sentiment_counts.get('Negative', 0)
    neutral_sentiment = sentiment_counts.get('Neutral', 0)

    if positive_sentiment > max(negative_sentiment, neutral_sentiment):
        sentiment_summary = "Overall sentiment is positive."
    elif negative_sentiment > max(positive_sentiment, neutral_sentiment):
        sentiment_summary = "Overall sentiment is negative."
    else:
        sentiment_summary = "Overall sentiment is neutral."

    # Combine summaries
    summary = f"{interest_summary} {sentiment_summary}"
    return summary

# Main function to run the analysis pipeline
def analyze_whatsapp_chat(input_path):
    # Parse chat text to DataFrame
    df = df_from_whatsapp(input_path)

    # Sentiment analysis using RoBERTa sentiment model
    sentiment_analyzer = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    def get_sentiment(text):
        result = sentiment_analyzer(str(text))[0]
        return result['label'], result['score']

    # Apply sentiment analysis to each message
    df[['sentiment', 'sentiment_score']] = df['message'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    # Interest detection with NLP keywords (optional)
    df['interest_level_nlp'] = df['message'].apply(detect_interest_nlp)

    # Train logistic regression model on NLP labels (optional)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['interest_level_nlp']
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    df['predicted_interest'] = clf.predict(vectorizer.transform(df['message']))

    # Print summary of sentiment counts
    sentiment_counts = df['sentiment'].value_counts()
    print("Overall Sentiment Distribution:", sentiment_counts.to_dict())

    # Return the DataFrame for further use if needed
    return df

# ---- Altered function to use Mistral API for feedback ----
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

    # Use OpenAI-compatible client for Mistral API
    client = openai.OpenAI(
        api_key=api_key_mistral,
        base_url="https://api.mistral.ai/v1"
    )

    response = client.chat.completions.create(
        model="mistral-large-latest",  # Or your available Mistral model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# ---- Main block ----
if __name__ == "__main__":
    api_key_mistral = "aKFEMuDwJOvtphHDDOrh2qbfRP7jEA1L"
    input_path = "/Users/madanmaskara/Documents/whatsapp-sales-analyzer/WhatsApp Chat with Meta AI.txt"
    df_result = analyze_whatsapp_chat(input_path)
    print(df_result[['username', 'message', 'sentiment', 'sentiment_score', 'interest_level_nlp', 'predicted_interest']].head())
    summary_output = generate_summary(df_result)
    print("\nSummary for Seller:")
    print(summary_output)

    seller_name = "Madan Maskara"  # <-- Change as needed
    feedback = get_seller_feedback(df_result, seller_name, api_key_mistral)
    print("\nSales Pitch Feedback for Seller:")
    print(feedback)
