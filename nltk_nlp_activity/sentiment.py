# ----------------------------------------------------
# GOAL:
# 1. Load movie reviews dataset
# 2. Convert text into features (unigrams)
# 3. Train a Naive Bayes classifier
# 4. Evaluate model accuracy
# 5. Display most informative words
# ----------------------------------------------------

# Import movie reviews dataset from NLTK
# This dataset contains positive and negative movie reviews
from nltk.corpus import movie_reviews

# Import random module to shuffle data
import random

# Import Naive Bayes classifier from NLTK
from nltk.classify import NaiveBayesClassifier

# Import accuracy function to evaluate the classifier
from nltk.classify.util import accuracy


# ----------------------------------------------------
# STEP 1: PREPARE THE DATASET
# ----------------------------------------------------

# Create a list of tuples:
# (list of words in the review, review category)
# category is either 'pos' (positive) or 'neg' (negative)
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Shuffle the dataset to avoid bias
# This ensures training and testing data are mixed randomly
random.shuffle(documents)


# ----------------------------------------------------
# STEP 2: FEATURE EXTRACTION (UNIGRAM MODEL)
# ----------------------------------------------------

# This function converts a document (list of words)
# into a dictionary of features
# Each word becomes a feature with value True
def document_features(words):

    # Example output:
    # {'great': True, 'movie': True, 'acting': True}
    return {word: True for word in words}


# Convert all documents into feature sets
# Each item: (feature_dictionary, category)
featuresets = [
    (document_features(d), c)
    for (d, c) in documents
]


# ----------------------------------------------------
# STEP 3: SPLIT INTO TRAINING AND TEST SETS
# ----------------------------------------------------

# First 1500 reviews for training
train_set = featuresets[:1500]

# Remaining reviews for testing
test_set = featuresets[1500:]


# ----------------------------------------------------
# STEP 4: TRAIN THE NAIVE BAYES CLASSIFIER
# ----------------------------------------------------

# Train the classifier using the training set
classifier = NaiveBayesClassifier.train(train_set)


# ----------------------------------------------------
# STEP 5: EVALUATE THE MODEL
# ----------------------------------------------------

# Calculate and print classification accuracy
print("Accuracy:", accuracy(classifier, test_set))


# ----------------------------------------------------
# STEP 6: DISPLAY MOST INFORMATIVE FEATURES
# ----------------------------------------------------

# Show the top 10 words that best distinguish
# positive vs negative reviews
classifier.show_most_informative_features(10)
