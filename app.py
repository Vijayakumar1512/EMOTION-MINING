import streamlit as st
import joblib
import pandas as pd
# Import Libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stopwords
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
svm_model = joblib.load('models/svm_model.pkl')
vectorizer = joblib.load('data/vectorizer.pkl')

# Define a function for preprocessing text
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit app
st.title("Sentiment Analysis")

review = st.text_area("Enter a product review:")

if st.button("Predict Sentiment"):
    if not review:
        st.error("Please enter a review.")
    else:
        # Preprocess the review
        processed_review = preprocess_text(review)
        features = vectorizer.transform([processed_review])

        # Predict sentiment
        prediction = svm_model.predict(features)

        # Get the score from the prediction (assuming your model outputs a sentiment score)
        sentiment_score = prediction[0]

        # Determine sentiment based on the score
        if sentiment_score < 2.5:
            sentiment = 'Negative'
        elif 2.5 <= sentiment_score <= 3.5:
            sentiment = 'Neutral'
        else:
            sentiment = 'Positive'

        st.write(f"Predicted Sentiment: {sentiment}")
