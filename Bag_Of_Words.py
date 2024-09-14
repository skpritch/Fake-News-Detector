"""This Fake News Detection model uses basic bag-of-words algorithm without word vectors"""
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

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
    description = ''
  return description

#Scrape website descriptions
def scrape_description(url):
  if not url.startswith('http'):
    url = 'http://' + url
  response = requests.get(url, timeout=10)
  html = response.text
  description = get_description_from_html(html)
  return description

def get_descriptions_from_data(data):
  # A dictionary mapping from url to description for the websites in 
  # train_data.
  descriptions = []
  for site in tqdm(data):
    descriptions.append(get_description_from_html(site[1]))
  return descriptions
  
#Look at both discriptions and expanded domain names
train_descriptions = get_descriptions_from_data(train_data)
train_urls = [url for (url, html, label) in train_data]

val_descriptions = get_descriptions_from_data(val_data)

vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
  X = vectorizer.transform(descriptions).todense()
  return X

print('\nPreparing train data...')
bow_train_X = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_train_y = [label for url, html, label in train_data]

print('\nPreparing val data...')
bow_val_X = vectorize_data_descriptions(val_descriptions, vectorizer)
bow_val_y = [label for url, html, label in val_data]

#Fit a model and print benchmarks
model = LogisticRegression(max_iter=1000)
model.fit(bow_train_X, bow_train_y)

train_y_pred = model.predict(bow_train_X)
print('Train accuracy: ', accuracy_score(bow_train_y, train_y_pred))

val_y_pred = model.predict(bow_val_X)
print('Val accuracy: ', accuracy_score(bow_val_y, val_y_pred))
print()

print(' Val confusion matrix: ')
print(confusion_matrix(bow_val_y, val_y_pred))
print()

prf = precision_recall_fscore_support(bow_val_y, val_y_pred)

print('Precision: ', prf[0][1])
print('Recall: ', prf[1][1])
print('F-Score: ', prf[2][1])