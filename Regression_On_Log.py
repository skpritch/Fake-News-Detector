"""This file contains a model based on logistic regression on the log of word counts to prevent extreme values"""

import math
import os
import numpy as np

import pickle

import requests, io, zipfile

# Download class resources...
r = requests.get("https://www.dropbox.com/s/2pj07qip0ei09xt/inspirit_fake_news_resources.zip?dl=1")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

basepath = '.'

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def prepare_data(data, featurizer):
    X = []
    y = []
    for datapoint in data:
        url, html, label = datapoint
        # Convert all text in HTML to lowercase, so <p>Hello.</p> is mapped to
        # <p>hello</p>.
        html = html.lower() 
        y.append(label)

        features = featurizer(url, html)

        # Gets the keys of the dictionary as descriptions, gets the values
        # as the numerical features.
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
    keywords = ["sports", "finance", "hillary", "free", "limited", "trump", "business", "trick", "obama", "easy", "neat"]
    
    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)
    
    return features

train_X, train_y, feature_descriptions = prepare_data(train_data, keyword_featurizer)
val_X, val_y, feature_descriptions = prepare_data(val_data, keyword_featurizer)

print('Number of features per example:', len(train_X[0]))
print('Feature descriptions:')
print(feature_descriptions)
print()
  
baseline_model = LogisticRegression(max_iter=1000)

#Train model on data and print benchmarks
baseline_model.fit(train_X, train_y)

train_y_pred = baseline_model.predict(train_X)
print('Train accuracy', accuracy_score(train_y, train_y_pred))

val_y_pred = baseline_model.predict(val_X)
print('Val accuracy', accuracy_score(val_y, val_y_pred))
print()

print('Confusion matrix:')
print(confusion_matrix(val_y, val_y_pred))

prf = precision_recall_fscore_support(val_y, val_y_pred)

print('Precision:', prf[0][1])
print('Recall:', prf[1][1])
print('F-Score:', prf[2][1])