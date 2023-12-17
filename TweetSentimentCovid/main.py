# %% [code]
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
# ## Loading Necessary libraries 😃

# %% [code]
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
import keras
from keras.models import Sequential
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

# %% [markdown]
# ## Reading the dataset 😁

# %% [code]
# /kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv
# /kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv

train_data = pd.read_csv('/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv',
                        encoding='latin_1')
test_data = pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv",encoding='latin_1')

# %% [code]
train_data.head()

# %% [code]
test_data.head()

# %% [markdown]
# ## Check for any null value/s  😶

# %% [code]
# check for null value in train_data
sns.heatmap(train_data.isnull());

# %% [code]
# check for null values in test data
sns.heatmap(test_data.isnull());

# %% [markdown]
# ## drop duplicate and 'NA' value/s 🙄

# %% [code]
train_data.drop_duplicates(inplace= True)
test_data.drop_duplicates(inplace=True)


# %% [code]
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# %% [markdown]
# ## Making checkpoint 🧐

# %% [code]
# copy the dataset into new data
train_df = train_data.copy()
test_df = test_data.copy()

# %% [code]
train_df.head()

# %% [markdown]
# ## check if any null value exists? 🤨

# %% [code]
print(train_df.isnull().sum())
print("*"*50)
print(test_df.isnull().sum())

# %% [code]
train_data.columns

# %% [markdown]
# ## performing small EDA 🤩

# %% [code]
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(11,4)})

# %% [code]
sns.countplot(train_df['Sentiment'])

# %% [code]
train_df.shape, test_df.shape

# %% [markdown]
# **As we can see we have 5 types of sentiment (but we can make extremely positive into positive and extremely negative into negative )**  😎

# %% [code]
def change_sen(sentiment):
    if sentiment == "Extremely Positive":
        return 'positive'
    elif sentiment == "Extremely Negative":
        return 'negative'
    elif sentiment == "Positive":
        return 'positive'
    elif sentiment == "Negative":
        return 'negative'
    else:
        return 'netural'

# %% [markdown]
# ### Applying the change_sen function/method  😝 

# %% [code]
train_df['Sentiment'] = train_df['Sentiment'].apply(lambda x: change_sen(x))
test_df['Sentiment'] = test_df['Sentiment'].apply(lambda x: change_sen(x))

# %% [code]
sns.countplot(train_df['Sentiment'])

# %% [code]
sns.countplot(test_df['Sentiment'])

# %% [markdown]
# ### Now time for "Data Clearning and processing"  🤯 

# %% [code]
# load stop words
stop_word = stopwords.words('english')

# %% [code]
def clean(text):

    #     remove urls
    text = re.sub(r'http\S+', " ", text)

    #     remove mentions
    text = re.sub(r'@\w+',' ',text)

    #     remove hastags
    text = re.sub(r'#\w+', ' ', text)

    #     remove digits
    text = re.sub(r'\d+', ' ', text)

    #     remove html tags
    text = re.sub('r<.*?>',' ', text)
    
    #     remove stop words 
    text = text.split()
    text = " ".join([word for word in text if not word in stop_word])
    
      
    return text

# %% [code]
train_df['OriginalTweet'] = train_df['OriginalTweet'].apply(lambda x: clean(x))
test_df['OriginalTweet'] = test_df['OriginalTweet'].apply(lambda x: clean(x))

# %% [code]
train_df.head()

# %% [markdown]
# ### We only need "OriginalTweet" and "Sentiment"
# 
#  😬 
# 
# ##### so only taking these columns

# %% [code]
df_train = train_df.iloc[:,4:]
df_test = test_df.iloc[:,4:]

# %% [code]
df_train.head()

# %% [markdown]
# ### now mapping the sentiment  🤠 
# - 0: Netural 
# - 1: Positive
# - 2: Negative

# %% [code]
l = {"netural":0, "positive":1,"negative":2}

# %% [code]
df_train['Sentiment'] = df_train['Sentiment'].map(l)
df_test['Sentiment']  = df_test['Sentiment'].map(l)

# %% [code]
df_train.head()

# %% [code]
x_train = df_train['OriginalTweet'].copy()
x_test = df_test['OriginalTweet'].copy()

y_train = df_train['Sentiment'].copy()
y_test = df_test['Sentiment'].copy()

# %% [code]
x_train.shape, y_train.shape,x_test.shape, y_test.shape

# %% [markdown]
# #### Maxiumn lenght of sequence  😧 

# %% [code]
max_len = np.max(x_train.apply(lambda x :len(x)))

# %% [code]
max_len

# %% [markdown]
# ### Tokenizer initlization  🤑 

# %% [code]
tokenizer = Tokenizer()

# %% [code]
tokenizer.fit_on_texts(x_train)
vocab_length = len(tokenizer.word_index) + 1

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

# %% [code]
print("Vocab length:", vocab_length)
print("Max sequence length:", max_len)

# %% [code]
embedding_dim = 16

# %% [markdown]
# ## Model Creation  😍 

# %% [code]
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_length, embedding_dim, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation='softmax')
])
# opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

# %% [code]
print(model.summary())

# %% [code]
tf.keras.utils.plot_model(model)

# %% [code]
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %% [code]
from keras.utils import to_categorical

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

# %% [markdown]
# ## Model training 🥱 😴

# %% [code]
num_epochs = 10
history = model.fit(x_train, y_train, epochs=num_epochs, 
                    validation_data=(x_test, y_test))

# %% [code]
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# %% [markdown]
# ## Model Accuracy and loss  😵 

# %% [code]
print(f"Accuracy on training data is:- {acc[-1]*100} %")
print(f"Loss {loss[-1]*100}")

print(f"Accuracy on validation data is:- {val_acc[-1]*100} %")
print(f"Loss {val_loss[-1]*100}")


# %% [markdown]
# ## Plotting  🤪 

# %% [code]
epochs = range(len(acc))

plt.plot(epochs, acc,'b',label='training acc')
plt.plot(epochs, val_acc, 'r', label='validation acc')
plt.legend()
plt.show()


plt.plot(epochs, loss,'b',label='training loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.legend()
plt.show()


# %% [code]
pred = model.predict_classes(x_test)

# %% [markdown]
# ## Confusion Matrix  🤔 

# %% [code]
cm = confusion_matrix(np.argmax(y_test,1),pred)
cm

# %% [code]
sns.heatmap(cm,annot=True)

# %% [markdown]
# ## Classification Report  🤫 

# %% [code]
print(classification_report(np.argmax(y_test,1),pred))

# %% [markdown]
# ## if you've learned something from this kernal then please "UPVOTE"  🤭 

# %% [markdown]
# ## Thanks for Watching!!  🙏 
# 
# - Happy Learning 😍 

# %% [code]
