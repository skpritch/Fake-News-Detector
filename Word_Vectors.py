"""This Fake News Detection model uses vectorizing words"""
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

VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

# Returns word vector for word if it exists, else return None.
def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None
    
    #@title Word Similarity { run: "auto", display-mode: "both" }

def cosine_similarity(vec1, vec2):    
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

word1 = "six" #@param {type:"string"}
word2 = "seven" #@param {type:"string"}

#Calculate cosine similarities between words
def cosine_similarity_of_words(word1, word2):
  vec1 = get_word_vector(word1)
  vec2 = get_word_vector(word2)
  
  if vec1 is None:
    print(word1, 'is not a valid word. Try another.')
  if vec2 is None:
    print(word2, 'is not a valid word. Try another.')
  if vec1 is None or vec2 is None:
    return None
  
  return cosine_similarity(vec1, vec2)
  

print('\nCosine similarity:', cosine_similarity_of_words(word1, word2))

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

        # divide the sum by the number of words added, so we have the
        # average word vector.
        if found_words > 0:
            X[i] /= found_words
            
    return X
  
glove_train_X = glove_transform_data_descriptions(train_descriptions)
glove_train_y = [label for (url, html, label) in train_data]

glove_val_X = glove_transform_data_descriptions(val_descriptions)
glove_val_y = [label for (url, html, label) in val_data]

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

train_and_evaluate_model(glove_train_X, glove_train_y, glove_val_X, glove_val_y)

