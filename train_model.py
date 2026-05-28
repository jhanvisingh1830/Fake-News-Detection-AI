import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# download stopwords
nltk.download('stopwords')

# stemmer
stemmer = PorterStemmer()

# stopwords
stop_words = stopwords.words('english')

# load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# smaller dataset for speed
fake = fake.head(5000)
true = true.head(5000)

# labels
fake["label"] = 0
true["label"] = 1

# combine
data = pd.concat([fake, true])

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

# clean data
data["title"] = data["title"].apply(clean_text)
data["text"] = data["text"].apply(clean_text)

# combine columns
data["content"] = data["title"] + " " + data["text"]

# input/output
x = data["content"]
y = data["label"]

# vectorizer
vectorizer = TfidfVectorizer(max_df=0.7)

x = vectorizer.fit_transform(x)

# split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# model
model = MultinomialNB()

# train
model.fit(x_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))

# save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")