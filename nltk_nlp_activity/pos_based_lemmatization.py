# ----------------------------------------------------
# GOAL:
# 1. Tokenize a sentence into words
# 2. Assign Part-of-Speech (POS) tags to each word
# 3. Convert POS tags to WordNet format
# 4. Lemmatize each word using its correct POS
# ----------------------------------------------------

# Import the NLTK library
# NLTK provides tools for Natural Language Processing
import nltk

# Import WordNetLemmatizer
# Lemmatizer reduces words to their base (dictionary) form
from nltk.stem import WordNetLemmatizer

# Import WordNet corpus
# WordNet defines grammatical categories like noun, verb, adjective, adverb
from nltk.corpus import wordnet


# ----------------------------------------------------
# STEP 1: CREATE A LEMMATIZER OBJECT
# ----------------------------------------------------

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


# ----------------------------------------------------
# STEP 2: MAP POS TAGS TO WORDNET TAGS
# ----------------------------------------------------

# NLTK POS tags (like NN, VB, JJ) are different
# from WordNet POS tags (NOUN, VERB, ADJ, ADV)
# This function converts NLTK tags to WordNet tags

def get_wordnet_post(tag):

    # If the POS tag starts with "J"
    # It represents an adjective (JJ, JJR, JJS)
    if tag.startswith("J"):
        return wordnet.ADJ

    # If the POS tag starts with "V"
    # It represents a verb (VB, VBD, VBG, etc.)
    elif tag.startswith("V"):
        return wordnet.VERB

    # If the POS tag starts with "N"
    # It represents a noun (NN, NNS, NNP)
    elif tag.startswith("N"):
        return wordnet.NOUN

    # If the POS tag starts with "R"
    # It represents an adverb (RB, RBR, RBS)
    elif tag.startswith("R"):
        return wordnet.ADV

    # Default case:
    # If POS tag is unknown, treat the word as a noun
    return wordnet.NOUN


# ----------------------------------------------------
# STEP 3: INPUT TEXT
# ----------------------------------------------------

# Example sentence for lemmatization
paragraph = "The striped bats are hanging on their feet for best."


# ----------------------------------------------------
# STEP 4: TOKENIZATION
# ----------------------------------------------------

# Split the sentence into individual words
tokens = nltk.word_tokenize(paragraph)


# ----------------------------------------------------
# STEP 5: POS TAGGING
# ----------------------------------------------------

# Assign Part-of-Speech tags to each word
# Example output: [('striped', 'JJ'), ('bats', 'NNS'), ...]
pos_tags = nltk.pos_tag(tokens)


# ----------------------------------------------------
# STEP 6: LEMMATIZATION WITH POS
# ----------------------------------------------------

# Lemmatize each word using:
# - the word itself
# - its converted WordNet POS tag
lemmatized_words = [
    lemmatizer.lemmatize(word, get_wordnet_post(tag))
    for word, tag in pos_tags
]


# ----------------------------------------------------
# STEP 7: OUTPUT
# ----------------------------------------------------

# Print the lemmatized words
print(lemmatized_words)
