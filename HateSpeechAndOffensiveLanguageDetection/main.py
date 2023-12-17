#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
#import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK resources
#nltk.download('stopwords')
#nltk.download('vader_lexicon')
#nltk.download('all')

sns.set(style="darkgrid")

# Data Exploration

data = pd.read_csv('./dataSet/train.csv')

print(data.head())
print(data['class'].value_counts())
print(data.describe())
print(data.isna().sum())
print(data.duplicated().sum())

fdata = data[['tweet', 'class']]

# Data Processing

# URL Removal
fdata["tweet"] = fdata["tweet"].apply(lambda x: re.sub(r'https?://\S+', ' ', str(x)))

# Lowercase Removal
fdata['tweet'] = fdata['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Punctuation Removal
fdata['tweet'] = fdata['tweet'].str.replace('[^\w\s]', '')

# Username/Handles Removal
fdata["tweet"] = fdata["tweet"].apply(lambda x: re.sub(r'@\w+', '', str(x)))

# Emoji Removal
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

fdata["tweet"] = fdata["tweet"].apply(str)
fdata["tweet"] = fdata["tweet"].apply(remove_emoji)

# Single character and double space removal
fdata["tweet"] = fdata["tweet"].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))
fdata["tweet"] = fdata["tweet"].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))

# Remove words "rt" and colons ":"
def remove_words_and_colons(text):
    words_to_remove = ["rt"]
    for word in words_to_remove:
        text = text.replace(word, "")
    text = text.replace(":", "")
    return text

fdata['tweet'] = fdata['tweet'].apply(remove_words_and_colons)

# Lambda function to remove special and numeric characters
remove_special_and_numeric = lambda text: re.sub(r"[^a-zA-Z\s]+", "", text)
fdata['tweet'] = fdata['tweet'].apply(remove_special_and_numeric)

# Stopword Removal
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

fdata["Text_stop"] = fdata["tweet"].apply(lambda text: remove_stopwords(text))

# Feature Engineering

# Tokenization of data
def tokenization(text):
    text = re.split('\W+', text)
    return text

fdata['Text_tokenized'] = fdata['Text_stop'].apply(lambda x: tokenization(x.lower()))

# Lemmatization of Data
nltk.download('wordnet')
wordNet = WordNetLemmatizer()
def lemmatizer(text):
    text = [wordNet.lemmatize(word) for word in text]
    return text

fdata['Text_lemmatized'] = fdata['Text_tokenized'].apply(lambda x: lemmatizer(x))

# Data Modeling

X = fdata["Text_stop"]
y = fdata["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Performing TF-IDF Conversion
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit((X_train).values.astype('U'))
print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))

X_train = vectoriser.transform((X_train).values.astype('U'))
X_test  = vectoriser.transform((X_test).values.astype('U'))

# Model Evaluation

def model_evaluate(model, X_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm,
                     index=['hate speech', 'offensive language', 'Neutral'],
                     columns=['hate speech', 'offensive language', 'Neutral'])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, cmap="Oranges", linecolor="gray")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()

# Logistic Regression Model
lr_model = LogisticRegression(C=1, max_iter=1000, penalty='l2', n_jobs=-1)
lr_model.fit(X_train, y_train)

model_evaluate(lr_model, X_test)

# Decision Tree Model
dtmodel = DecisionTreeClassifier()
dtc = dtmodel.fit(X_train, y_train)
model_evaluate(dtc, X_test)

# K-Neighbors Classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
model_evaluate(neigh, X_test)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit
