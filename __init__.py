# Creating our text classifier for sentiment analysis
# Is the message positive or negative?
# Binary classifier

import nltk
from nltk.stem import PorterStemmer
import random
from nltk.corpus import movie_reviews, stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

stopwords = set(stopwords.words('english'))
punctuation = '.\'\"-,;:?!_--[]()'
ps = PorterStemmer()

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        word_list = []
        for word in list(movie_reviews.words(fileid)):
            if word not in stopwords and word not in punctuation:
                word_list.append(ps.stem(word))
        documents.append((np.array(word_list), category))

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

train_set, test_set = train_test_split(documents)

print(train_set[:10])


def find_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
