"""Model 3: Logistic Regression with Domain and Keyword Features"""

import os  # To handle file paths
import math  # For mathematical operations
import pickle  # To load data
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # Evaluation metrics

# Define the path to the fake news data folder
data_folder = '/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Fake News Detector/fake_news_data'

# Function to prepare data for training and validation
def prepare_data(data, featurizer):
    X = []  # Store features
    y = []  # Store labels
    for datapoint in data:
        url, html, label = datapoint  # Unpack the datapoint
        html = html.lower()  # Convert HTML to lowercase
        y.append(label)  # Append label (real or fake)
        features = featurizer(url, html)  # Extract features
        feature_descriptions, feature_values = zip(*features.items())  # Extract feature descriptions and values
        X.append(feature_values)  # Store feature values
    return X, y, feature_descriptions  # Return features, labels, and descriptions

# Function to compute the log count of a keyword in HTML
def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))  # Log-transform the count for normalization

# Function to extract both domain-based and keyword-based features
def combined_featurizer(url, html):
    features = {}
    
    # Domain features
    features['.com domain'] = url.endswith('.com')
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')

    # Keyword features
    keywords = ["sports", "finance", "hillary", "free", "limited", "trump", "business", "trick", "obama", "easy", "neat"]
    for keyword in keywords:
        features[keyword + ' keyword'] = get_normalized_count(html, keyword)  # Add log-normalized counts of keywords
    
    return features  # Return combined feature dictionary

# Function to train the logistic regression model
def train_model(train_X, train_y, val_X, val_y):
    model = LogisticRegression(solver='liblinear')  # Initialize logistic regression model with 'liblinear' solver
    model.fit(train_X, train_y)  # Train the model
    return model  # Return trained model

# Function to train and evaluate the model
def train_and_evaluate_model(train_X, train_y, val_X, val_y):
    model = train_model(train_X, train_y, val_X, val_y)  # Train the model

    # Predict and evaluate on training data
    train_y_pred = model.predict(train_X)
    print('Train accuracy: ', accuracy_score(train_y, train_y_pred))  # Output training accuracy

    # Predict and evaluate on validation data
    val_y_pred = model.predict(val_X)
    print('Val accuracy: ', accuracy_score(val_y, val_y_pred))  # Output validation accuracy
    print()

    # Print confusion matrix for validation data
    print(' Val confusion matrix: ')
    print(confusion_matrix(val_y, val_y_pred))
    print()

    # Compute precision, recall, and F1-Score for validation data
    prf = precision_recall_fscore_support(val_y, val_y_pred)
    print('Precision: ', prf[0][1])
    print('Recall: ', prf[1][1])
    print('F1-Score: ', prf[2][1])
    print()

    return model  # Return trained model

# Load the training and validation data from the "fake_news_data" folder
train_val_data_path = os.path.join(data_folder, 'train_val_data.pkl')
with open(train_val_data_path, 'rb') as f:
    train_data, val_data = pickle.load(f)

# Prepare features and labels for training and validation sets
train_X, train_y, feature_descriptions = prepare_data(train_data, combined_featurizer)
val_X, val_y, feature_descriptions = prepare_data(val_data, combined_featurizer)

# Train and evaluate the model
train_and_evaluate_model(train_X, train_y, val_X, val_y)
