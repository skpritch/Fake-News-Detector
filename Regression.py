"""This file contains a model based on logistic regression"""

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
  
# Returns a dictionary mapping from plaintext feature descriptions to numerical
# features for a (url, html) pair.
def domain_featurizer(url, html):
    features = {}
    
    # Binary features for the domain name extension.
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
    
    return features

#Prepare storage for training and testing data
with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)
  
print('Number of train examples:', len(train_data))
print('Number of val examples:', len(val_data))

train_X, train_y, feature_descriptions = prepare_data(train_data, domain_featurizer)
val_X, val_y, feature_descriptions = prepare_data(val_data, domain_featurizer)

print('Number of features per example:', len(train_X[0]))
print('Feature descriptions:')
print(feature_descriptions)

baseline_model = LogisticRegression()
baseline_model.fit(train_X, train_y)

#Run and test model; print benchmarks
train_y_pred = baseline_model.predict(train_X)
print('Train accuracy', accuracy_score(train_y, train_y_pred))

val_y_pred = baseline_model.predict(val_X)
print('Val accuracy', accuracy_score(val_y, val_y_pred))

print('Confusion matrix:')
print(confusion_matrix(val_y, val_y_pred))

prf = precision_recall_fscore_support(val_y, val_y_pred)

print('Precision:', prf[0][1])
print('Recall:', prf[1][1])
print('F-Score:', prf[2][1])

