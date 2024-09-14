"""This Fake News Detection model combines BOW and GloVe for additional accuracy"""
import math
import os
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe

import pickle

import requests, io, zipfile
r = requests.get("https://www.dropbox.com/s/2pj07qip0ei09xt/inspirit_fake_news_resources.zip?dl=1")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

basepath = '.'

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)
  
print('Number of train examples:', len(train_data))
print('Number of val examples:', len(val_data))



def get_descriptions_from_data(data):
  # A dictionary mapping from url to description for the websites in 
  # train_data.
  descriptions = []
  for site in tqdm(data):
    ### YOUR CODE HERE ###
    descriptions.append(get_description_from_html(site[1]))
    ### END CODE ###
  return descriptions
  

train_descriptions = get_descriptions_from_data(train_data)
train_urls = [url for (url, html, label) in train_data]
val_descriptions = get_descriptions_from_data(val_data)




#Helper function to fit a model and evaluate conserved across methods
def train_model(train_X, train_y, val_X, val_y):
  model = LogisticRegression(solver='liblinear')
  model.fit(train_X, train_y)
  
  return model

def train_and_evaluate_model(train_X, train_y, val_X, val_y):
  model = train_model(train_X, train_y, val_X, val_y)
  train_y_pred = model.predict(train_X)
  print('Train accuracy: ', accuracy_score(train_y, train_y_pred))

  val_y_pred = model.predict(val_X)
  print('Val accuracy: ', accuracy_score(val_y, val_y_pred))
  print()

  print(' Val confusion matrix: ')
  print(confusion_matrix(val_y, val_y_pred))
  print()

  prf = precision_recall_fscore_support(val_y, val_y_pred)

  print('Precision: ', prf[0][1])
  print('Recall: ', prf[1][1])
  print('F1-Score: ', prf[2][1])
  print()
  
  return model




def prepare_data(data, featurizer):
    X = []
    y = []
    for datapoint in data:
        url, html, label = datapoint
        # convert all text in HTML to lowercase, so <p>Hello.</p> is mapped to
        # <p>hello</p>. This will help us later when we extract features from 
        # the HTML, as we will be able to rely on the HTML being lowercase.
        html = html.lower() 
        y.append(label)

        features = featurizer(url, html)

        # Gets the keys of the dictionary as descriptions, gets the values
        # as the numerical features. Don't worry about exactly what zip does!
        feature_descriptions, feature_values = zip(*features.items())

        X.append(feature_values)

    return X, y, feature_descriptions
  
# Gets the log count of a phrase/keyword in HTML (transforming the phrase/keyword
# to lowercase).
def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))

# Returns a dictionary mapping from plaintext feature descriptions to numerical
# features for a (url, html) pair.
def keyword_featurizer(url, html):
    features = {}
    
    # Same as before.
    features['.com domain'] = url.endswith('.com')
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.org domain'] = url.endswith('.org')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')
    
    keywords = ['trump', 'biden', 'clinton', 'sports', 'finance', 'limited', 'free']
    
    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)
    
    return features

keyword_train_X, train_y, feature_descriptions = prepare_data(train_data, keyword_featurizer)
keyword_val_X, val_y, feature_descriptions = prepare_data(val_data, keyword_featurizer)



vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(data_descriptions, vectorizer):
  X = vectorizer.transform(data_descriptions).todense()
  return X

bow_train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
train_y = [label for url, html, label in train_data]

bow_val_X = vectorize_data_descriptions(val_descriptions, vectorizer)
val_y = [label for url, html, label in val_data]



#Prepare Word Vector Data
VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split(): 
            vec = get_word_vector(word)
            if vec is not None:
                # Increment found_words and add vec to X[i].
                found_words += 1
                X[i] += vec

        # We divide the sum by the number of words added, so we have the
        # average word vector.
        if found_words > 0:
            X[i] /= found_words
            
    return X

glove_train_X = glove_transform_data_descriptions(train_descriptions)

glove_val_X = glove_transform_data_descriptions(val_descriptions)



#Combine 
def combine_features(X_list):
  return np.concatenate(X_list, axis=1)

# First, produce combined_train_X and combined_val_X using 2 calls to 
# combine_features, using keyword_train_X, bow_train_X, glove_train_X
# and keyword_val_X, bow_val_X, glove_val_X from before.

combined_train_X = combine_features([keyword_train_X, bow_train_X, glove_train_X])
combined_val_X = combine_features([keyword_val_X, bow_val_X, glove_val_X])

model = train_and_evaluate_model(combined_train_X, train_y, combined_val_X, val_y)