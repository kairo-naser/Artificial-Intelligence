# ----------------------------------------------------
# GOAL:
# 1. Tokenize text into individual words
# 2. Assign Part-of-Speech (POS) tags
# 3. Select only nouns (candidate topic words)
# 4. Count how often each noun appears
# 5. Display the most frequent nouns
# ----------------------------------------------------

# Import the NLTK library
# Used for tokenization and POS tagging
import nltk

# Import Counter for frequency analysis
# Counter counts how many times each word appears
from collections import Counter


# ----------------------------------------------------
# STEP 1: INPUT TEXT
# ----------------------------------------------------

# Sample text where we want to extract topic words
text = "Artificial intelligence and machine learning are transforming industries worldwide."


# ----------------------------------------------------
# STEP 2: TOKENIZATION
# ----------------------------------------------------

# Split the text into individual tokens (words)
# Example output:
# ['Artificial', 'intelligence', 'and', 'machine', 'learning', ...]
tokens = nltk.word_tokenize(text)


# ----------------------------------------------------
# STEP 3: PART-OF-SPEECH TAGGING
# ----------------------------------------------------

# Assign POS tags to each token
# Example:
# ('Artificial', 'JJ'), ('intelligence', 'NN')
pos_tags = nltk.pos_tag(tokens)


# ----------------------------------------------------
# STEP 4: SELECT NOUNS ONLY
# ----------------------------------------------------

# Keep only words whose POS tag starts with 'NN'
# NN   → noun
# NNS  → plural noun
# NNP  → proper noun
# NNPS → plural proper noun
nouns = [
    word for word, tag in pos_tags
    if tag.startswith('NN')
]


# ----------------------------------------------------
# STEP 5: FREQUENCY ANALYSIS
# ----------------------------------------------------

# Count how often each noun appears
freq = Counter(nouns)


# ----------------------------------------------------
# STEP 6: OUTPUT
# ----------------------------------------------------

# Print the 10 most common nouns (topic candidates)
# Example output:
# [('intelligence', 1), ('learning', 1), ('industries', 1)]
print(freq.most_common(10))
