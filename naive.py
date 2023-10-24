import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Convert text data to numerical features
vectorizer = TfidfVectorizer()
train['keyword'].fillna('', inplace=True)
X_train = vectorizer.fit_transform(train['keyword'])
y_train = train['target']

# Preprocess test data
test['keyword'].fillna('', inplace=True)

# Convert test data to numerical features
X_test = vectorizer.transform(test['keyword'])

# Train machine learning model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict target values for test data
y_pred = clf.predict(X_test)

# Save predictions to a CSV file # Changed to for submission2
submission_naive = pd.DataFrame({'id': test['id'], 'target': y_pred})
submission_naive.to_csv('submission_naive.csv', index=False)