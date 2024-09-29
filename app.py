# Streamlit configuration (must be called as the first Streamlit command)
import streamlit as st

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

# Import necessary libraries
import joblib
import numpy as np
from PIL import Image

# Load your pre-trained model and vectorizer
try:
    model = joblib.load('sentiment_model.pkl')  # Path to your saved model
    vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Path to your saved TfidfVectorizer
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

# Streamlit App Interface

# Add a header and description
st.title('Sentiment Analysis App')
st.write("""
This app uses Natural Language Processing (NLP) techniques to classify the sentiment of a given text as *Positive* or *Negative*. 
Simply input a sentence or paragraph, and the model will predict the sentiment for you.
""")

# Optional: Add an image or logo (if you have one)
# image = Image.open('your_logo.png')
# st.image(image, caption='Sentiment Analysis App')

# Input text box for user to type or paste text
st.subheader("Input Text")
user_input = st.text_area("Enter the text you want to analyze for sentiment:")

# Button for prediction
if st.button("Analyze"):
    if user_input:
        try:
            # Preprocess the input and make predictions
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)
            
            # Show the result to the user
            st.subheader("Sentiment Result:")
            if prediction == 1:  # Assuming 1 is Positive sentiment
                st.success("😊 Positive Sentiment Detected!")
            else:  # Assuming 0 is Negative sentiment
                st.error("😞 Negative Sentiment Detected.")
            
            # Optionally, display the input text again for reference
            st.write("You entered:")
            st.write(user_input)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text to analyze.")

# Add footer with contact or additional info
st.write("---")
st.write("Created by [Your Name]. For more information, visit [Your Website].")
