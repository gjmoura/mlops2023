#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing Libraries

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import string

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('all')

from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator

from collections import Counter

from matplotlib import ticker
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


sns.set(style="darkgrid")


# # Data Exploration

# In[ ]:


data=pd.read_csv('/kaggle/input/hate-speech-and-offensive-language-detection/train.csv')
data.head()


# In[ ]:


data['class'].value_counts()


# Here,
# 0 - hate speech
# 1 - offensive language
# 2 - neither

# In[ ]:


data.describe()


# In[ ]:


data.isna().sum()


# There are no null values in the dataset

# In[ ]:


data.duplicated().sum()


# There are no duplicate values in the dataset

# Therefore there is no requirement for data cleaning

# In[ ]:


fdata=data[['tweet','class']]
fdata.head()


# ### Data Visualization - Distribution of tweet counts per classification category

# In[ ]:


sns.histplot(data['class'])


# 

# # Data Processing

# To remove the links from text attribute, we have defined the following function with the help of 'Regex'.

# ### URL Removal

# In[ ]:


# URL Removal
fdata["tweet"] = fdata["tweet"].apply(lambda x:re.sub(r'https?://\S+', '', str(x)))
fdata["tweet"]


# ### Lowercase Removal

# In[ ]:


# Lowercase removal
fdata['tweet'] = fdata['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
fdata['tweet']


# ### Punctuation Removal

# In[ ]:


# Punctuation Removal

fdata['tweet'] = fdata['tweet'].str.replace('[^\w\s]','')
fdata['tweet']


# ### Username/Handles Removal

# In[ ]:


# Function to remove usernames/handles

fdata["tweet"] = fdata["tweet"].apply(lambda x:re.sub(r'@\w+', '', str(x)))
fdata["tweet"]


# 

# ### Emoji Removal

# In[ ]:


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
fdata["tweet"]


# ### Single character and double space removal

# In[ ]:


# Single character and double space removal
fdata["tweet"] = fdata["tweet"].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
fdata["tweet"] = fdata["tweet"].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
fdata["tweet"]


# Lets see the top 10 most common words in the dataset

# In[ ]:


# Most common words
from collections import Counter
cnt = Counter()
for text in fdata["tweet"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)


# The words 'rt' and the colon are most common and are not needed for our analysis. We will remove them

# In[ ]:


# Function to remove words "rt" and colons ":"
def remove_words_and_colons(text):
    words_to_remove = ["rt"]
    for word in words_to_remove:
        text = text.replace(word, "")
    text = text.replace(":", "")
    return text

# Applying the function to the 'tweet' column
fdata['tweet'] = fdata['tweet'].apply(remove_words_and_colons)

# Displaying the updated DataFrame
fdata


# There are a lot of numeric and special characters in our tweets. We will go ahead and remove them

# In[ ]:


# Lambda function to remove special and numeric characters
remove_special_and_numeric = lambda text: re.sub(r"[^a-zA-Z\s]+", "", text)

# Apply the function to the 'tweets' column using lambda
fdata['tweet'] = fdata['tweet'].apply(remove_special_and_numeric)

fdata.head()


# ### Stopword Removal

# In[ ]:


# Stopword Removal
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

fdata["Text_stop"] = fdata["tweet"].apply(lambda text: remove_stopwords(text))
fdata


# # Feature Engineering

# ## Tokenization of data

# In[ ]:


import textblob           
from textblob import TextBlob

def tokenization(text):
    text = re.split('\W+', text)
    return text

fdata['Text_tokenized'] = fdata['Text_stop'].apply(lambda x: tokenization(x.lower()))
fdata.head()


# ### Lemmitization of Data

# In[ ]:


get_ipython().system(' python3 -m nltk.downloader wordnet')
get_ipython().system(' unzip /root/nltk_data/corpora/wordnet.zip -d /root/nltk_data/corpora/')
get_ipython().system('unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/')


# In[ ]:


nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
wordNet = WordNetLemmatizer()
def lemmatizer(text):
    text = [wordNet.lemmatize(word) for word in text]
    return text

fdata['Text_lemmatized'] = fdata['Text_tokenized'].apply(lambda x: lemmatizer(x))
fdata.head()


# # Data Modeling

# In[ ]:


X = fdata["Text_stop"]
y = fdata["class"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)


# ### Performing TF-IDF Conversion

# In[ ]:


vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit((X_train).values.astype('U'))
print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))


# In[ ]:


X_train = vectoriser.transform((X_train).values.astype('U'))
X_test  = vectoriser.transform((X_test).values.astype('U'))


# # Model Evaluation

# In[ ]:


def model_evaluate(model,X_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm,
                     index = ['hate speech','offensive language','Neutral'], 
                     columns = ['hate speech','offensive language','Neutral'])

    #Plotting the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, cmap="Oranges",linecolor="gray")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()


# ## Logistic Regression Model

# In[ ]:


lr_model = LogisticRegression(C = 1, max_iter = 1000, penalty = 'l2', n_jobs=-1)
lr_model.fit(X_train  ,y_train)


# In[ ]:


model_evaluate(lr_model ,X_test)


# ## Decision Tree Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier 

dtmodel = DecisionTreeClassifier()
dtc = dtmodel.fit(X_train,y_train)


# In[ ]:


model_evaluate(dtc ,X_test)


# ## K-Neighbors Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train  ,y_train)


# In[ ]:


model_evaluate(neigh ,X_test)


# ## Random Forest Classifier

# In[ ]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rfc=RandomForestClassifier(n_estimators=10)

#Train the model using the training sets y_pred=clf.predict(X_test)
rfc.fit(X_train,y_train)


# In[ ]:


model_evaluate(rfc ,X_test)


# In[ ]:


#Import XGB FModel
import xgboost as xgb

#Create a XGB Classifier
xgb_model=xgb.XGBClassifier(objective="multi:softprob")

#Train the model using the training sets
xgb_model.fit(X_train,y_train)


# In[ ]:


model_evaluate(xgb_model,X_test)


# ## LGBM Classifier

# In[ ]:


import lightgbm as lgb
lgb = lgb.LGBMClassifier()
lgb.fit(X_train,y_train)


# In[ ]:


model_evaluate(lgb,X_test)


# The Logistic Regression,XGB Classifier and LGBM Classifer perform better than the other models. We will select the LGBM Model and perform Hyperparameter to try to improve the metrics for our model

# # Hyperparameter Tuning

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

# Define the LightGBM model
model = lgb.LGBMClassifier()

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                   n_iter=8, scoring='accuracy', cv=3, random_state=42)


# In[ ]:


# Fit the model
random_search.fit(X_train, y_train)


# In[ ]:


# Print the best parameters and the best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)


# In[ ]:


# Evaluate the model with best parameters on the test set
best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test Score:", test_score)


# In[ ]:


model_evaluate(best_model,X_test)
