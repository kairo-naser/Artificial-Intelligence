# ----------------------------------------------------
# GOAL:
# 1. Split text into individual words (tokenization)
# 2. Convert all text to lowercase (normalization)
# 3. Remove punctuation symbols
# 4. Remove common English stopwords (like "the", "is")
# 5. Count word frequencies
# 6. Display the top 20 most frequent words
# ----------------------------------------------------

# Import the Natural Language Toolkit (NLTK) library
# NLTK provides tools for text processing and NLP tasks
import nltk

# Import the word_tokenize function
# This function splits text into individual words
from nltk.tokenize import word_tokenize

# Import stopwords list from NLTK
# Stopwords are common words that usually carry little meaning
from nltk.corpus import stopwords

# Import Counter to count word frequency
from collections import Counter

# Import string module
# Used here to access punctuation characters like . , ! ?
import string


# ----------------------------------------------------
# INPUT TEXT
# This is the raw text we want to analyze
# ----------------------------------------------------
text = """
Artificial intelligence continues transforming industries in 2025.
Companies like OpenAI, Google, and Meta are releasing new models
that enhance productivity and reshape the global economy.
Governments in the United States and Europe are debating regulations
to ensure safe deployment of advanced AI systems.
"""


# ----------------------------------------------------
# STEP 1: TOKENIZATION
# ----------------------------------------------------

# Convert the entire text to lowercase
# This ensures words like "AI" and "ai" are treated the same
text = text.lower()

# Split the text into words (tokens)
# Example output: ["artificial", "intelligence", "continues", ...]
tokens = word_tokenize(text)


# ----------------------------------------------------
# STEP 2: REMOVE STOPWORDS AND PUNCTUATION
# ----------------------------------------------------

# Load English stopwords from NLTK
# Examples: "the", "is", "are", "in", "to"
stop_words = set(stopwords.words('english'))

# Create a cleaned list of words
# Conditions:
# - word is NOT a stopword
# - word is NOT punctuation (.,!? etc.)
clean_text = [
    token for token in tokens
    if token not in stop_words
    and token not in string.punctuation
]


# ----------------------------------------------------
# STEP 3: FREQUENCY COUNT
# ----------------------------------------------------

# Count how many times each word appears
# Counter creates a dictionary-like object
freq = Counter(clean_text)


# ----------------------------------------------------
# STEP 4: DISPLAY RESULTS
# ----------------------------------------------------

# Print the top 20 most frequent words
# Output format: [('word', count), ('word', count), ...]
print(freq.most_common(20))
