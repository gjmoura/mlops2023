# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:25.616155Z","iopub.execute_input":"2023-12-15T05:34:25.616596Z","iopub.status.idle":"2023-12-15T05:34:26.194825Z","shell.execute_reply.started":"2023-12-15T05:34:25.616564Z","shell.execute_reply":"2023-12-15T05:34:26.193595Z"}}
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

# %% [markdown]
# # Importing Libraries

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:26.197034Z","iopub.execute_input":"2023-12-15T05:34:26.19757Z","iopub.status.idle":"2023-12-15T05:34:49.576799Z","shell.execute_reply.started":"2023-12-15T05:34:26.197535Z","shell.execute_reply":"2023-12-15T05:34:49.574735Z"}}
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
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

# %% [markdown]
# # Data Exploration  

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:49.578354Z","iopub.execute_input":"2023-12-15T05:34:49.580292Z","iopub.status.idle":"2023-12-15T05:34:49.746155Z","shell.execute_reply.started":"2023-12-15T05:34:49.580254Z","shell.execute_reply":"2023-12-15T05:34:49.744509Z"}}
data=pd.read_csv('/kaggle/input/hate-speech-and-offensive-language-detection/train.csv')
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:49.749339Z","iopub.execute_input":"2023-12-15T05:34:49.749796Z","iopub.status.idle":"2023-12-15T05:34:53.914609Z","shell.execute_reply.started":"2023-12-15T05:34:49.749753Z","shell.execute_reply":"2023-12-15T05:34:53.913344Z"}}
data['class'].value_counts()

# %% [markdown]
# Here,
# 0 - hate speech
# 1 - offensive language
# 2 - neither

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:53.916925Z","iopub.execute_input":"2023-12-15T05:34:53.917564Z","iopub.status.idle":"2023-12-15T05:34:53.999979Z","shell.execute_reply.started":"2023-12-15T05:34:53.917532Z","shell.execute_reply":"2023-12-15T05:34:53.998613Z"}}
data.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:54.001804Z","iopub.execute_input":"2023-12-15T05:34:54.002518Z","iopub.status.idle":"2023-12-15T05:34:54.018295Z","shell.execute_reply.started":"2023-12-15T05:34:54.002476Z","shell.execute_reply":"2023-12-15T05:34:54.016419Z"}}
data.isna().sum()

# %% [markdown]
# There are no null values in the dataset

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:54.020716Z","iopub.execute_input":"2023-12-15T05:34:54.022186Z","iopub.status.idle":"2023-12-15T05:34:54.10155Z","shell.execute_reply.started":"2023-12-15T05:34:54.02214Z","shell.execute_reply":"2023-12-15T05:34:54.100387Z"}}
data.duplicated().sum()

# %% [markdown]
# There are no duplicate values in the dataset

# %% [markdown]
# Therefore there is no requirement for data cleaning

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:54.103839Z","iopub.execute_input":"2023-12-15T05:34:54.105145Z","iopub.status.idle":"2023-12-15T05:34:54.136569Z","shell.execute_reply.started":"2023-12-15T05:34:54.105097Z","shell.execute_reply":"2023-12-15T05:34:54.134825Z"}}
fdata=data[['tweet','class']]
fdata.head()

# %% [markdown]
# ### Data Visualization - Distribution of tweet counts per classification category

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T06:11:00.606727Z","iopub.execute_input":"2023-12-15T06:11:00.607178Z","iopub.status.idle":"2023-12-15T06:11:01.109261Z","shell.execute_reply.started":"2023-12-15T06:11:00.607145Z","shell.execute_reply":"2023-12-15T06:11:01.107975Z"}}
sns.histplot(data['class'])

# %% [markdown]
# 

# %% [markdown]
# # Data Processing

# %% [markdown]
# To remove the links from text attribute, we have defined the following function with the help of 'Regex'.

# %% [markdown]
# ### URL Removal

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:54.139504Z","iopub.execute_input":"2023-12-15T05:34:54.141641Z","iopub.status.idle":"2023-12-15T05:34:54.246123Z","shell.execute_reply.started":"2023-12-15T05:34:54.141567Z","shell.execute_reply":"2023-12-15T05:34:54.245141Z"}}
# URL Removal
fdata["tweet"] = fdata["tweet"].apply(lambda x:re.sub(r'https?://\S+', '', str(x)))
fdata["tweet"]

# %% [markdown]
# ### Lowercase Removal

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:54.250165Z","iopub.execute_input":"2023-12-15T05:34:54.251435Z","iopub.status.idle":"2023-12-15T05:34:54.399683Z","shell.execute_reply.started":"2023-12-15T05:34:54.251399Z","shell.execute_reply":"2023-12-15T05:34:54.398567Z"}}
# Lowercase removal
fdata['tweet'] = fdata['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
fdata['tweet']

# %% [markdown]
# ### Punctuation Removal

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:54.401367Z","iopub.execute_input":"2023-12-15T05:34:54.402071Z","iopub.status.idle":"2023-12-15T05:34:54.43692Z","shell.execute_reply.started":"2023-12-15T05:34:54.40203Z","shell.execute_reply":"2023-12-15T05:34:54.4358Z"}}
# Punctuation Removal

fdata['tweet'] = fdata['tweet'].str.replace('[^\w\s]','')
fdata['tweet']

# %% [markdown]
# ### Username/Handles Removal

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:54.438426Z","iopub.execute_input":"2023-12-15T05:34:54.439569Z","iopub.status.idle":"2023-12-15T05:34:54.653453Z","shell.execute_reply.started":"2023-12-15T05:34:54.4395Z","shell.execute_reply":"2023-12-15T05:34:54.652369Z"}}
# Function to remove usernames/handles

fdata["tweet"] = fdata["tweet"].apply(lambda x:re.sub(r'@\w+', '', str(x)))
fdata["tweet"]

# %% [markdown]
# 

# %% [markdown]
# ### Emoji Removal

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:54.654972Z","iopub.execute_input":"2023-12-15T05:34:54.656035Z","iopub.status.idle":"2023-12-15T05:34:54.900733Z","shell.execute_reply.started":"2023-12-15T05:34:54.655994Z","shell.execute_reply":"2023-12-15T05:34:54.899679Z"}}
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

# %% [markdown]
# ### Single character and double space removal

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:54.902067Z","iopub.execute_input":"2023-12-15T05:34:54.902401Z","iopub.status.idle":"2023-12-15T05:34:55.311601Z","shell.execute_reply.started":"2023-12-15T05:34:54.90237Z","shell.execute_reply":"2023-12-15T05:34:55.310238Z"}}
# Single character and double space removal
fdata["tweet"] = fdata["tweet"].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
fdata["tweet"] = fdata["tweet"].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
fdata["tweet"]

# %% [markdown]
# Lets see the top 10 most common words in the dataset

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:55.313383Z","iopub.execute_input":"2023-12-15T05:34:55.313937Z","iopub.status.idle":"2023-12-15T05:34:55.554648Z","shell.execute_reply.started":"2023-12-15T05:34:55.313904Z","shell.execute_reply":"2023-12-15T05:34:55.553228Z"}}
# Most common words
from collections import Counter
cnt = Counter()
for text in fdata["tweet"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)

# %% [markdown]
# The words 'rt' and the colon are most common and are not needed for our analysis. We will remove them

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:55.556398Z","iopub.execute_input":"2023-12-15T05:34:55.556841Z","iopub.status.idle":"2023-12-15T05:34:55.603215Z","shell.execute_reply.started":"2023-12-15T05:34:55.556805Z","shell.execute_reply":"2023-12-15T05:34:55.601838Z"}}
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

# %% [markdown]
# There are a lot of numeric and special characters in our tweets. We will go ahead and remove them

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:55.60497Z","iopub.execute_input":"2023-12-15T05:34:55.605389Z","iopub.status.idle":"2023-12-15T05:34:55.78033Z","shell.execute_reply.started":"2023-12-15T05:34:55.605354Z","shell.execute_reply":"2023-12-15T05:34:55.778902Z"}}
# Lambda function to remove special and numeric characters
remove_special_and_numeric = lambda text: re.sub(r"[^a-zA-Z\s]+", "", text)

# Apply the function to the 'tweets' column using lambda
fdata['tweet'] = fdata['tweet'].apply(remove_special_and_numeric)

fdata.head()

# %% [markdown]
# ### Stopword Removal

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:55.782249Z","iopub.execute_input":"2023-12-15T05:34:55.78271Z","iopub.status.idle":"2023-12-15T05:34:55.915442Z","shell.execute_reply.started":"2023-12-15T05:34:55.782676Z","shell.execute_reply":"2023-12-15T05:34:55.913983Z"}}
# Stopword Removal
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

fdata["Text_stop"] = fdata["tweet"].apply(lambda text: remove_stopwords(text))
fdata

# %% [markdown]
# # Feature Engineering

# %% [markdown]
# ## Tokenization of data

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:34:55.917028Z","iopub.execute_input":"2023-12-15T05:34:55.91743Z","iopub.status.idle":"2023-12-15T05:34:56.177088Z","shell.execute_reply.started":"2023-12-15T05:34:55.917398Z","shell.execute_reply":"2023-12-15T05:34:56.175569Z"}}
import textblob           
from textblob import TextBlob

def tokenization(text):
    text = re.split('\W+', text)
    return text

fdata['Text_tokenized'] = fdata['Text_stop'].apply(lambda x: tokenization(x.lower()))
fdata.head()

# %% [markdown]
# ### Lemmitization of Data

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:40:43.553287Z","iopub.execute_input":"2023-12-15T05:40:43.553796Z","iopub.status.idle":"2023-12-15T05:40:48.672442Z","shell.execute_reply.started":"2023-12-15T05:40:43.553754Z","shell.execute_reply":"2023-12-15T05:40:48.670736Z"}}
! python3 -m nltk.downloader wordnet
! unzip /root/nltk_data/corpora/wordnet.zip -d /root/nltk_data/corpora/
!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:40:51.536082Z","iopub.execute_input":"2023-12-15T05:40:51.53659Z","iopub.status.idle":"2023-12-15T05:40:55.582908Z","shell.execute_reply.started":"2023-12-15T05:40:51.536543Z","shell.execute_reply":"2023-12-15T05:40:55.581785Z"}}
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
wordNet = WordNetLemmatizer()
def lemmatizer(text):
    text = [wordNet.lemmatize(word) for word in text]
    return text

fdata['Text_lemmatized'] = fdata['Text_tokenized'].apply(lambda x: lemmatizer(x))
fdata.head()

# %% [markdown]
# # Data Modeling

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:44:07.637884Z","iopub.execute_input":"2023-12-15T05:44:07.638363Z","iopub.status.idle":"2023-12-15T05:44:07.651473Z","shell.execute_reply.started":"2023-12-15T05:44:07.638331Z","shell.execute_reply":"2023-12-15T05:44:07.650518Z"}}
X = fdata["Text_stop"]
y = fdata["class"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)

# %% [markdown]
# ### Performing TF-IDF Conversion

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:45:01.390908Z","iopub.execute_input":"2023-12-15T05:45:01.39134Z","iopub.status.idle":"2023-12-15T05:45:02.680458Z","shell.execute_reply.started":"2023-12-15T05:45:01.391308Z","shell.execute_reply":"2023-12-15T05:45:02.679267Z"}}
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit((X_train).values.astype('U'))
print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:45:14.470243Z","iopub.execute_input":"2023-12-15T05:45:14.470723Z","iopub.status.idle":"2023-12-15T05:45:15.263256Z","shell.execute_reply.started":"2023-12-15T05:45:14.470685Z","shell.execute_reply":"2023-12-15T05:45:15.26205Z"}}
X_train = vectoriser.transform((X_train).values.astype('U'))
X_test  = vectoriser.transform((X_test).values.astype('U'))

# %% [markdown]
# # Model Evaluation

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:47:34.121184Z","iopub.execute_input":"2023-12-15T05:47:34.122308Z","iopub.status.idle":"2023-12-15T05:47:34.130723Z","shell.execute_reply.started":"2023-12-15T05:47:34.122245Z","shell.execute_reply":"2023-12-15T05:47:34.129655Z"}}
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

# %% [markdown]
# ## Logistic Regression Model

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:48:15.803537Z","iopub.execute_input":"2023-12-15T05:48:15.803996Z","iopub.status.idle":"2023-12-15T05:48:27.315128Z","shell.execute_reply.started":"2023-12-15T05:48:15.803963Z","shell.execute_reply":"2023-12-15T05:48:27.313729Z"}}
lr_model = LogisticRegression(C = 1, max_iter = 1000, penalty = 'l2', n_jobs=-1)
lr_model.fit(X_train  ,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:48:29.890873Z","iopub.execute_input":"2023-12-15T05:48:29.892109Z","iopub.status.idle":"2023-12-15T05:48:30.343338Z","shell.execute_reply.started":"2023-12-15T05:48:29.892047Z","shell.execute_reply":"2023-12-15T05:48:30.341868Z"}}
model_evaluate(lr_model ,X_test)

# %% [markdown]
# ## Decision Tree Model

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:52:40.017251Z","iopub.execute_input":"2023-12-15T05:52:40.017724Z","iopub.status.idle":"2023-12-15T05:52:57.24244Z","shell.execute_reply.started":"2023-12-15T05:52:40.017692Z","shell.execute_reply":"2023-12-15T05:52:57.241135Z"}}
from sklearn.tree import DecisionTreeClassifier 

dtmodel = DecisionTreeClassifier()
dtc = dtmodel.fit(X_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:54:09.533186Z","iopub.execute_input":"2023-12-15T05:54:09.533762Z","iopub.status.idle":"2023-12-15T05:54:09.98289Z","shell.execute_reply.started":"2023-12-15T05:54:09.533721Z","shell.execute_reply":"2023-12-15T05:54:09.981478Z"}}
model_evaluate(dtc ,X_test)

# %% [markdown]
# ## K-Neighbors Classifier

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:55:31.815246Z","iopub.execute_input":"2023-12-15T05:55:31.815773Z","iopub.status.idle":"2023-12-15T05:55:31.831396Z","shell.execute_reply.started":"2023-12-15T05:55:31.815716Z","shell.execute_reply":"2023-12-15T05:55:31.82972Z"}}
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train  ,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:55:41.286941Z","iopub.execute_input":"2023-12-15T05:55:41.287375Z","iopub.status.idle":"2023-12-15T05:55:50.010004Z","shell.execute_reply.started":"2023-12-15T05:55:41.287345Z","shell.execute_reply":"2023-12-15T05:55:50.008656Z"}}
model_evaluate(neigh ,X_test)

# %% [markdown]
# ## Random Forest Classifier

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:56:22.172457Z","iopub.execute_input":"2023-12-15T05:56:22.173003Z","iopub.status.idle":"2023-12-15T05:56:33.791988Z","shell.execute_reply.started":"2023-12-15T05:56:22.172964Z","shell.execute_reply":"2023-12-15T05:56:33.7906Z"}}
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rfc=RandomForestClassifier(n_estimators=10)

#Train the model using the training sets y_pred=clf.predict(X_test)
rfc.fit(X_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T05:56:43.723902Z","iopub.execute_input":"2023-12-15T05:56:43.724327Z","iopub.status.idle":"2023-12-15T05:56:44.252677Z","shell.execute_reply.started":"2023-12-15T05:56:43.724295Z","shell.execute_reply":"2023-12-15T05:56:44.251835Z"}}
model_evaluate(rfc ,X_test)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T06:01:30.276312Z","iopub.execute_input":"2023-12-15T06:01:30.276807Z","iopub.status.idle":"2023-12-15T06:02:07.862009Z","shell.execute_reply.started":"2023-12-15T06:01:30.276775Z","shell.execute_reply":"2023-12-15T06:02:07.860335Z"}}
#Import XGB FModel
import xgboost as xgb

#Create a XGB Classifier
xgb_model=xgb.XGBClassifier(objective="multi:softprob")

#Train the model using the training sets
xgb_model.fit(X_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T06:02:36.613818Z","iopub.execute_input":"2023-12-15T06:02:36.614316Z","iopub.status.idle":"2023-12-15T06:02:37.157765Z","shell.execute_reply.started":"2023-12-15T06:02:36.614282Z","shell.execute_reply":"2023-12-15T06:02:37.156728Z"}}
model_evaluate(xgb_model,X_test)

# %% [markdown]
# ## LGBM Classifier

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T08:30:37.712968Z","iopub.execute_input":"2023-12-15T08:30:37.71344Z","iopub.status.idle":"2023-12-15T08:30:46.376675Z","shell.execute_reply.started":"2023-12-15T08:30:37.713407Z","shell.execute_reply":"2023-12-15T08:30:46.375378Z"}}
import lightgbm as lgb
lgb = lgb.LGBMClassifier()
lgb.fit(X_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T06:07:05.08905Z","iopub.execute_input":"2023-12-15T06:07:05.089517Z","iopub.status.idle":"2023-12-15T06:07:05.831406Z","shell.execute_reply.started":"2023-12-15T06:07:05.089484Z","shell.execute_reply":"2023-12-15T06:07:05.830492Z"}}
model_evaluate(lgb,X_test)

# %% [markdown]
# The Logistic Regression,XGB Classifier and LGBM Classifer perform better than the other models. We will select the LGBM Model and perform Hyperparameter to try to improve the metrics for our model

# %% [markdown]
# # Hyperparameter Tuning

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T08:46:10.267906Z","iopub.execute_input":"2023-12-15T08:46:10.268434Z","iopub.status.idle":"2023-12-15T08:46:10.277198Z","shell.execute_reply.started":"2023-12-15T08:46:10.268395Z","shell.execute_reply":"2023-12-15T08:46:10.275734Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T08:46:11.103032Z","iopub.execute_input":"2023-12-15T08:46:11.103507Z","iopub.status.idle":"2023-12-15T08:48:05.04838Z","shell.execute_reply.started":"2023-12-15T08:46:11.103474Z","shell.execute_reply":"2023-12-15T08:48:05.046709Z"}}
# Fit the model
random_search.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T08:48:38.777012Z","iopub.execute_input":"2023-12-15T08:48:38.77753Z","iopub.status.idle":"2023-12-15T08:48:38.785675Z","shell.execute_reply.started":"2023-12-15T08:48:38.777493Z","shell.execute_reply":"2023-12-15T08:48:38.783937Z"}}
# Print the best parameters and the best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T08:48:52.376454Z","iopub.execute_input":"2023-12-15T08:48:52.377004Z","iopub.status.idle":"2023-12-15T08:48:52.573325Z","shell.execute_reply.started":"2023-12-15T08:48:52.376947Z","shell.execute_reply":"2023-12-15T08:48:52.572354Z"}}
# Evaluate the model with best parameters on the test set
best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test Score:", test_score)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-15T08:49:08.988781Z","iopub.execute_input":"2023-12-15T08:49:08.990197Z","iopub.status.idle":"2023-12-15T08:49:09.669153Z","shell.execute_reply.started":"2023-12-15T08:49:08.990129Z","shell.execute_reply":"2023-12-15T08:49:09.668206Z"}}
model_evaluate(best_model,X_test)