"""import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Öffnen der corpus.txt-Datei zum Lesen
with open('corpus.txt', 'r') as file:
    # Leeren der Liste, die die Texte im Corpus enthalten wird
    corpus = []
    # Iterieren über jede Zeile in der corpus.txt-Datei
    for line in file:
        # Hinzufügen der Zeile zur Liste
        corpus.append(line)

# Erstellen eines Tokenizers, der die Texte im Corpus tokenisiert
tokenizer = Tokenizer()
# Fit des Tokenizers auf die Texte im Corpus
tokenizer.fit_on_texts(corpus)

# Konvertieren der Texte im Corpus in numerische Sequenzen
sequences = tokenizer.texts_to_sequences(corpus)
# Pad der Sequenzen auf die gleiche Länge
data = pad_sequences(sequences)

# Teilen der numerischen Sequenzen in Trainings- und Testdaten auf
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Erstellen eines CNN-Modells in TensorFlow
model = tf.keras.Sequential()
# Hinzufügen von Convolutional- und Pooling-Schichten zum Modell
model.add(tf.keras.layers.Conv1D(32, 3, activation='relu', input_)

#!pip install transformers
#!pip install biobert-model

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/BioBert-Base")"""
import bayes as bayes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from io import StringIO
import seaborn as sns

with open('../docsources/text4.txt') as file:
    Data = pd.read_csv(file)
Data = Data[['glucose', 'canser']]
Data = Data[pd.notnull(Data['patient'])]
Data.head()
encoder = preprocessing.LabelEncoder()
fig = plt.figure(figsize=(8,6))
Data.groupby('product').consumer_complaint_narrative.count()
plt.bar(ylim=0)
plt.show()
