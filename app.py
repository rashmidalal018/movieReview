# import pandas as pd
# import pickle as pk
# from nltk import NaiveBayesClassifier
# import streamlit as st

# model = pk.load(open('model.pkl','rb'))

# review = st.text_input('Enter Movie Review')

# if st.button('Predict'):
    

import pickle as pk
from nltk.classify import NaiveBayesClassifier
import streamlit as st

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pk.load(file)

# Function to preprocess input text
def clean_review(words):
    # Your cleaning code here
    pass

# Function to extract features from input text
def get_features_dict(words):
    # Your feature extraction code here
    pass

# Function to predict sentiment
def predict_sentiment(review_text):
    # Clean and preprocess the input text
    cleaned_review = clean_review(review_text)
    # Extract features
    features = get_features_dict(cleaned_review)
    if features:
        # Predict using the loaded model
        prediction = model.classify(features)
        return prediction
    else:
        return "Error: Unable to extract features from the input."

# Streamlit UI
st.title("Movie Review Sentiment Analysis")

# Input text area for user to enter movie review
review = st.text_area('Enter Movie Review')

# Button to trigger prediction
if st.button('Predict'):
    if review:
        # Perform prediction
        prediction = predict_sentiment(review)
        # Display the prediction result
        st.write(f"Predicted Sentiment: {prediction}")
    else:
        st.warning("Please enter a movie review to predict.")

