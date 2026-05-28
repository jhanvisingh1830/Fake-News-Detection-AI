from flask import Flask, render_template, request

import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# flask
app = Flask(__name__)

# nltk
nltk.download('stopwords')

# stemmer
stemmer = PorterStemmer()

# stopwords
stop_words = stopwords.words('english')

# load saved model
model = pickle.load(open("model.pkl", "rb"))

# load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# cleaning function
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# home page
@app.route('/')
def home():
    return render_template('index.html')

# prediction
@app.route('/predict', methods=['POST'])
def predict():

    news = request.form['news']

    cleaned_news = clean_text(news)

    vector_input = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector_input)

    probability = model.predict_proba(vector_input)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template(
        'index.html',
        prediction=result,
        confidence=confidence
    )

