# ----------------------------------------------------
# GOAL:
# 1. Tokenize text into words
# 2. Assign Part-of-Speech (POS) tags
# 3. Detect Named Entities using NLTK
# 4. Extract only PERSON and GPE entities
# ----------------------------------------------------

# Import the NLTK library
# NLTK provides tools for NLP tasks such as NER
import nltk

# Import ne_chunk function
# ne_chunk performs Named Entity Recognition
# It groups words into entities like PERSON, GPE, ORGANIZATION
from nltk import ne_chunk


# ----------------------------------------------------
# STEP 1: INPUT TEXT
# ----------------------------------------------------

# Example text containing people and locations
text = "Barack Obama was born in Hawaii. He was elected president of the USA."


# ----------------------------------------------------
# STEP 2: TOKENIZATION
# ----------------------------------------------------

# Split the text into individual words
# Example output: ['Barack', 'Obama', 'was', 'born', 'in', 'Hawaii', '.']
token = nltk.word_tokenize(text)


# ----------------------------------------------------
# STEP 3: PART-OF-SPEECH TAGGING
# ----------------------------------------------------

# Assign POS tags to each token
# Example: ('Barack', 'NNP'), ('Obama', 'NNP')
pos_tags = nltk.pos_tag(token)


# ----------------------------------------------------
# STEP 4: NAMED ENTITY RECOGNITION (NER)
# ----------------------------------------------------

# ne_chunk builds a tree structure
# It groups tokens into named entities
# Example labels: PERSON, GPE, ORGANIZATION
tree = ne_chunk(pos_tags)


# ----------------------------------------------------
# STEP 5: EXTRACT PERSON AND GPE ENTITIES
# ----------------------------------------------------

# Create an empty list to store extracted entities
entities = []

# Loop through each item in the NER tree
for subtree in tree:

    # Check if the item is a named entity subtree
    # Named entities have a "label" attribute
    if hasattr(subtree, 'label'):

        # Only keep PERSON and GPE entities
        if subtree.label() in ['PERSON', 'GPE']:

            # Combine words that form the entity
            # Example: ['Barack', 'Obama'] → "Barack Obama"
            entity = " ".join([leaf[0] for leaf in subtree.leaves()])

            # Store entity with its label
            entities.append((entity, subtree.label()))


# ----------------------------------------------------
# STEP 6: OUTPUT
# ----------------------------------------------------

# Print extracted named entities
# Example output:
# [('Barack Obama', 'PERSON'), ('Hawaii', 'GPE'), ('USA', 'GPE')]
print(entities)
