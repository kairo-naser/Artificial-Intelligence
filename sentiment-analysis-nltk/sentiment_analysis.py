# ============================================================
# SENTIMENT ANALYSIS USING NLTK (VADER)
# ============================================================
# This program analyzes the sentiment of a sentence
# and classifies it as:
# - Positive
# - Negative
# - Neutral
#
# It uses the VADER Sentiment Analyzer from NLTK,
# which is specially designed for short texts.
# ============================================================


# ------------------------------------------------------------
# STEP 0: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

# nltk:
# Natural Language Toolkit – used for text processing
import nltk

# SentimentIntensityAnalyzer:
# A rule-based sentiment analysis tool (VADER)
from nltk.sentiment import SentimentIntensityAnalyzer


# ------------------------------------------------------------
# STEP 1: DOWNLOAD REQUIRED LEXICON
# ------------------------------------------------------------
# VADER uses a predefined sentiment dictionary (lexicon)
# This download is required only ONCE per environment
nltk.download('vader_lexicon')


# ------------------------------------------------------------
# STEP 2: CREATE SENTIMENT ANALYZER OBJECT
# ------------------------------------------------------------
# This object will calculate sentiment scores
sia = SentimentIntensityAnalyzer()


# ------------------------------------------------------------
# STEP 3: INPUT TEXT DATA
# ------------------------------------------------------------
# A list of sentences to analyze
# (Can be expanded to multiple sentences)
sentence = ["My name is Ali."]


# ------------------------------------------------------------
# STEP 4: DEFINE SENTIMENT ANALYSIS FUNCTION
# ------------------------------------------------------------
def sentiment_analyzer_scores(text):
    """
    This function takes a sentence as input
    and returns its sentiment label:
    Positive, Negative, or Neutral
    """

    # polarity_scores returns a dictionary:
    # {
    #   'neg': negative score,
    #   'neu': neutral score,
    #   'pos': positive score,
    #   'compound': final combined score
    # }
    score = sia.polarity_scores(text)['compound']

    # Compound score interpretation (VADER standard):
    # score >=  0.05  → Positive
    # score <= -0.05  → Negative
    # otherwise       → Neutral
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# ------------------------------------------------------------
# STEP 5: CALL FUNCTION AND DISPLAY RESULT
# ------------------------------------------------------------

# Analyze the first sentence in the list
result = sentiment_analyzer_scores(sentence[0])

# Print the sentiment result
print("Sentence:", sentence[0])
print("Predicted Sentiment:", result)
