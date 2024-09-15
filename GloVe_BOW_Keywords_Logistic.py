"""Model 6: Combined Features (Keywords + Bag-of-Words + Word Vectors) with Logistic Regression"""

import os
import pickle
import numpy as np
from bs4 import BeautifulSoup as bs
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torchtext.vocab import GloVe
from joblib import Parallel, delayed

# Define the path to the fake news data folder
data_folder = '/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Fake News Detector/fake_news_data'

# Load GloVe word vectors
VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Securely extract meta descriptions from HTML with BeautifulSoup
def get_description_from_html(html):
    """
    Extracts meta description from HTML using BeautifulSoup. 
    Ensures the parsing is done securely and avoids code execution within the HTML.
    """
    soup = bs(html, "html.parser")
    description_tag = (soup.find('meta', attrs={'name':'og:description'}) or
                       soup.find('meta', attrs={'property':'description'}) or
                       soup.find('meta', attrs={'name':'description'}))
    if description_tag:
        description = description_tag.get('content') or ''
    else:
        description = ''  # Return empty string if no description is found
    return description

# Parallelized description extraction
def parallel_get_descriptions_from_data(data):
    """
    Parallelizes the process of extracting descriptions from data for efficiency.
    """
    return Parallel(n_jobs=-1)(delayed(get_description_from_html)(site[1]) for site in data)

# Helper functions for GloVe word vectors
def get_word_vector(word):
    """
    Securely fetches the GloVe word vector for a given word. Returns None if the word is not found.
    """
    try:
        return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
        return None

def glove_transform_data_descriptions(descriptions):
    """
    Transforms a list of descriptions into a matrix of averaged GloVe word vectors.
    """
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
                found_words += 1
                X[i] += vec
        if found_words > 0:
            X[i] /= found_words  # Average the vectors
    return X

# Function to train and evaluate the model using all CPU cores
def train_model(train_X, train_y, val_X, val_y):
    """
    Trains a logistic regression model using all available CPU cores.
    """
    model = LogisticRegression(max_iter=1000, n_jobs=-1)  # Use all available CPU cores
    model.fit(train_X, train_y)
    return model

def train_and_evaluate_model(train_X, train_y, val_X, val_y):
    """
    Trains and evaluates the model, reporting the accuracy and performance metrics.
    """
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
train_descriptions = parallel_get_descriptions_from_data(train_data)
val_descriptions = parallel_get_descriptions_from_data(val_data)

# Use CountVectorizer to create bag-of-words features
vectorizer = CountVectorizer(max_features=300)
vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
    """
    Vectorizes the descriptions using CountVectorizer and converts the result to a NumPy array.
    """
    return np.asarray(vectorizer.transform(descriptions).todense())

# Bag-of-Words feature extraction
bow_train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_val_X = vectorize_data_descriptions(val_descriptions, vectorizer)

# GloVe word vector transformation
glove_train_X = glove_transform_data_descriptions(train_descriptions)
glove_val_X = glove_transform_data_descriptions(val_descriptions)

# Combine features (Bag-of-Words + GloVe)
combined_train_X = np.concatenate([bow_train_X, glove_train_X], axis=1)
combined_val_X = np.concatenate([bow_val_X, glove_val_X], axis=1)

train_y = [label for url, html, label in train_data]
val_y = [label for url, html, label in val_data]

# Train and evaluate the model
train_and_evaluate_model(combined_train_X, train_y, combined_val_X, val_y)
