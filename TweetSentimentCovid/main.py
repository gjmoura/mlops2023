# Importing necessary libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import re
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from string import punctuation
import keras
from keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Loading the dataset
train_data = pd.read_csv('./dataSet/Corona_NLP_train.csv', encoding='latin_1')
test_data = pd.read_csv("./dataSet/Corona_NLP_test.csv", encoding='latin_1')

# Checking for null values
sns.heatmap(train_data.isnull());
sns.heatmap(test_data.isnull());

# Dropping duplicate and 'NA' values
train_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Creating checkpoint
train_df = train_data.copy()
test_df = test_data.copy()

# Applying sentiment changes
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
        return 'neutral'

train_df['Sentiment'] = train_df['Sentiment'].apply(lambda x: change_sen(x))
test_df['Sentiment'] = test_df['Sentiment'].apply(lambda x: change_sen(x))

# Data cleaning and processing
stop_word = stopwords.words('english')

def clean(text):
    text = re.sub(r'http\S+', " ", text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub('r<.*?>', ' ', text)
    text = text.split()
    text = " ".join([word for word in text if not word in stop_word])
    return text

train_df['OriginalTweet'] = train_df['OriginalTweet'].apply(lambda x: clean(x))
test_df['OriginalTweet'] = test_df['OriginalTweet'].apply(lambda x: clean(x))

# Selecting relevant columns
df_train = train_df.iloc[:, 4:]
df_test = test_df.iloc[:, 4:]

# Mapping sentiment
l = {"neutral": 0, "positive": 1, "negative": 2}
df_train['Sentiment'] = df_train['Sentiment'].map(l)
df_test['Sentiment'] = df_test['Sentiment'].map(l)

# Preparing data for model training
x_train = df_train['OriginalTweet'].copy()
x_test = df_test['OriginalTweet'].copy()
y_train = df_train['Sentiment'].copy()
y_test = df_test['Sentiment'].copy()

# Tokenization and padding
max_len = np.max(x_train.apply(lambda x: len(x)))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
vocab_length = len(tokenizer.word_index) + 1
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

# Model creation
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_length, embedding_dim, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Model training
num_epochs = 1
history = model.fit(x_train, to_categorical(y_train, 3), epochs=num_epochs, validation_data=(x_test, to_categorical(y_test, 3)))

# Model evaluation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Accuracy on training data is: {acc[-1] * 100} %")
print(f"Loss {loss[-1] * 100}")
print(f"Accuracy on validation data is: {val_acc[-1] * 100} %")
print(f"Loss {val_loss[-1] * 100}")

# Plotting
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='training acc')
plt.plot(epochs, val_acc, 'r', label='validation acc')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'b', label='training loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.legend()
plt.show()

# Confusion matrix
pred = model.predict_classes(x_test)
cm = confusion_matrix(np.argmax(to_categorical(y_test, 3), 1), pred)
sns.heatmap(cm, annot=True)

# Classification report
print(classification_report(np.argmax(to_categorical(y_test, 3), 1), pred))
