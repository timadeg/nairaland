import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
wn = nltk.WordNetLemmatizer()
import urllib
import re
from sklearn.preprocessing import LabelEncoder

# Title of the app
st.title('NAIRALAND NEWS CLASSIFICATION')

# Load model
model_optimize = tf.keras.models.load_model('C:/Users/User/Downloads/nairalandMODEL')

# Load the encoder
encoder_classes = np.load('C:/Users/User/Downloads/nairalandMODEL/encoder_classes.npy', allow_pickle=True)
encoder = LabelEncoder()
encoder.classes_ = encoder_classes

# Function to clean text
def clean_text(data):
    if pd.isna(data):
        return ""

    data = str(data)  # Convert data to string

    data = data.lower() # Convert data to lower case


    # Remove punctuations, HTML tags, URLs, and emojis
    punct_tag = re.compile(r'[^\w\s]')
    html_tag = re.compile(r'<.*?>')
    img_tag = re.compile(r'\[img\].*?\[/img\]')
    additional_tags = re.compile(r'\[(font|size|center|s|url|b|i)(=.*?)?\].*?\[/\1\]')
    emoji_clean = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)

    data = punct_tag.sub(r'', data)
    data = html_tag.sub(r'', data)
    data = img_tag.sub(r'', data)
    data = additional_tags.sub(r'', data)
    data = ' '.join([word for word in data.split() if not (urllib.parse.urlsplit(word).scheme or word.startswith('www.'))])
    data = emoji_clean.sub(r'', data)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    additional_stop_words = {'dey', 'na', 'will', 'said', 'wey', 'and', 'to', 'of'}
    stop_words.update(additional_stop_words)
    words = data.split()
    filtered_words = [word for word in words if word not in stop_words]

    # Lemmatization
    wn = WordNetLemmatizer()
    lemmatized_words = [wn.lemmatize(w) for w in filtered_words]
    data = ' '.join(lemmatized_words)

    return data

# Take user input
article_title = st.text_input('Enter the news article title')
article_body = st.text_input('Enter the news article body')

# Predict button
if st.button('PREDICT CATEGORY'):
    news_article = article_title + ' ' + article_body

    # Preprocessing the input text using the clean_text function
    preprocessed_article = clean_text(news_article)

    # Converting the preprocessed text into a NumPy array
    input_data = np.array([preprocessed_article])

    # Getting the predicted probabilities for each category
    predicted_probabilities = model_optimize.predict(input_data)

    # Finding the class with the highest probability
    predicted_class = np.argmax(predicted_probabilities, axis=1)

    # Getting the corresponding category using the label encoder
    predicted_category = encoder.inverse_transform(predicted_class)

    st.write("Predicted Category:", predicted_category[0])
