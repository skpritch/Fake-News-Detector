"""Model 4: Logistic Regression with Bag of Words"""

import os
import pickle
import numpy as np
from bs4 import BeautifulSoup as bs
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Define the path to the fake news data folder
data_folder = '/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Fake News Detector/fake_news_data'

# Function to extract meta descriptions from HTML
def get_description_from_html(html):
    soup = bs(html, "html.parser")
    description_tag = (soup.find('meta', attrs={'name':'og:description'}) or
                       soup.find('meta', attrs={'property':'description'}) or
                       soup.find('meta', attrs={'name':'description'}))
    if description_tag:
        description = description_tag.get('content') or ''
    else:
        description = ''  # Return empty string if no description is found
    return description

# Function to train and evaluate the model using all CPU cores
def train_model(train_X, train_y, val_X, val_y):
    model = LogisticRegression(max_iter=1000, n_jobs=-1)  # Use all available CPU cores
    model.fit(train_X, train_y)
    return model

def train_and_evaluate_model(train_X, train_y, val_X, val_y):
    model = train_model(train_X, train_y, val_X, val_y)
    
    # Evaluate on training data
    train_y_pred = model.predict(train_X)
    print('Train accuracy: ', accuracy_score(train_y, train_y_pred))
    
    # Evaluate on validation data
    val_y_pred = model.predict(val_X)
    print('Val accuracy: ', accuracy_score(val_y, val_y_pred))
    print()
    
    print('Val confusion matrix:')
    print(confusion_matrix(val_y, val_y_pred))
    print()
    
    prf = precision_recall_fscore_support(val_y, val_y_pred)
    print('Precision: ', prf[0][1])
    print('Recall: ', prf[1][1])
    print('F1-Score: ', prf[2][1])
    print()
    
    return model

# Load the training and validation data
train_val_data_path = os.path.join(data_folder, 'train_val_data.pkl')
with open(train_val_data_path, 'rb') as f:
    train_data, val_data = pickle.load(f)

# Parallelized description extraction
from joblib import Parallel, delayed

def parallel_get_descriptions_from_data(data):
    return Parallel(n_jobs=-1)(delayed(get_description_from_html)(site[1]) for site in data)

train_descriptions = parallel_get_descriptions_from_data(train_data)
val_descriptions = parallel_get_descriptions_from_data(val_data)

# Use CountVectorizer to create bag-of-words features
vectorizer = CountVectorizer(max_features=300)
vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
    return np.asarray(vectorizer.transform(descriptions).todense())

# Vectorize the descriptions for training and validation
bow_train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_val_X = vectorize_data_descriptions(val_descriptions, vectorizer)

# Slice the labels to match the reduced data size
bow_train_y = [label for url, html, label in train_data]  # First 100 labels
bow_val_y = [label for url, html, label in val_data]  # First 100 validation labels

# Train and evaluate the model
train_and_evaluate_model(bow_train_X, bow_train_y, bow_val_X, bow_val_y)
