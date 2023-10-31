# Kaggle NLP Competition: Disaster Tweet Classification

Welcome to my solution for the Kaggle "Getting Started" competition on Natural Language Processing (NLP) with Disaster Tweets. In this competition, the goal is to build a machine learning model that predicts whether a given tweet is about a real disaster or not.

## Overview

This repository contains my solution to the competition, the code, and the results of my solution. 

## Dataset

The dataset for this competition consists of 10,000 tweets that have been manually labeled as either describing a real disaster (target=1) or not (target=0). The dataset may contain text that is considered profane or offensive.

You can access the dataset on Kaggle's competition page [here](https://www.kaggle.com/competitions/nlp-getting-started).

## Approach

My approach to solving this problem involves various NLP techniques and machine learning algorithms. Here's a high-level overview of the steps I took:

1. **Data Preprocessing:** I cleaned and preprocessed the text data, which included tasks like removing special characters, tokenization.

2. **Model Selection:** I experimented with several machine learning models, including but not limited to Logistic Regression, Random Forest, and Gradient Boosting.

3. **Model Evaluation:** I used the F1 score as the evaluation metric since it is specified in the competition guidelines.


## Results
My final submission achieved an F1 score of 0.79895 on the test dataset. You can find more details in the competition's [leaderboard](https://www.kaggle.com/competitions/nlp-getting-started/leaderboard).
