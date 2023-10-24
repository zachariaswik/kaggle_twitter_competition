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

# Preprocess text data
stop_words = set(stopwords.words('english'))
train['text'] = train['text'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_words]))
test['text'] = test['text'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_words]))

# Convert text data to numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['text'])
y_train = train['target']
X_test = vectorizer.transform(test['text'])

# Train machine learning model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict target values for test data
y_pred = clf.predict(X_test)

# Save predictions to a CSV file
submission = pd.DataFrame({'id': test['id'], 'target': y_pred})
submission.to_csv('submission.csv', index=False)



