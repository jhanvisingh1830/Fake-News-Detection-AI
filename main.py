import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier

from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
# download stopwords
nltk.download('stopwords')

# load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# add labels
fake["label"] = 0
true["label"] = 1

# combine datasets
data = pd.concat([fake, true])

# shuffle dataset
data = data.sample(frac=1)

# stopwords list
stop_words = stopwords.words('english')

# text cleaning function
def clean_text(text):

    # convert to lowercase
    text = text.lower()

    # remove brackets text
    text = re.sub(r'\[.*?\]', '', text)

    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # split into words
    words = text.split()

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    # join words
    text = " ".join(words)

    return text

# apply cleaning function
data["title"] = data["title"].apply(clean_text)
data["text"] = data["text"].apply(clean_text)

data["content"] = data["title"] + " " + data["text"]
# input and output
x = data["content"]
y = data["label"]

# convert text to vectors
vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(x)

# split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# create model
model = MultinomialNB()

# train model
model.fit(x_train, y_train)

# predictions
y_pred = model.predict(x_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
# custom prediction

news = input("Enter news text: ")

# clean input
news = clean_text(news)

# convert to vector
news_vector = vectorizer.transform([news])

# predict
prediction = model.predict(news_vector)
probability = model.predict_proba(vector_input)
confidence = round(max(probability[0]) * 100, 2)
# output
if prediction[0] == 0:
    print("Fake News")
else:
    print("Real News")