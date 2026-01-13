🧠 NLP Activities Using NLTK (Python)

This repository demonstrates five core Natural Language Processing (NLP) tasks implemented using the Natural Language Toolkit (NLTK).
Each task addresses a specific requirement from the activity “Use NLTK” and is implemented as a separate Python file.

All implementations rely on standard, well-documented NLP techniques and are suitable for academic submission and practical learning.

📌 Library used: NLTK (Natural Language Toolkit)
📖 Official documentation: https://www.nltk.org/

📂 Activity Overview
Activity: Use NLTK

The repository answers the following five tasks:

Tokenization & Cleaning

POS-Based Lemmatization

Named Entity Recognition (NER)

Sentiment Classification

Topic Word Identification

Each section below explains what the file does, how it works, and why it is correct.

1️⃣ Tokenization & Cleaning

File: tokenization.py

✅ Task

Given a news article, produce a cleaned token list and top 20 frequent words (excluding stopwords).

🔍 What This File Does

Converts text to lowercase

Tokenizes the text into words

Removes:

Stopwords (e.g., the, is, and)

Punctuation symbols

Counts word frequency

Displays the top 20 most frequent meaningful words

🧠 Why This Works

Tokenization and cleaning are fundamental preprocessing steps in NLP.
Removing stopwords and punctuation improves signal quality and helps identify important content words.

📚 Sources:

NLTK Tokenization: https://www.nltk.org/api/nltk.tokenize.html

Stopwords in NLP: https://www.nltk.org/book/ch02.html

Text preprocessing (Stanford NLP): https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html

2️⃣ POS-Based Lemmatization

File: pos_based_lemmatization.py

✅ Task

Lemmatize a paragraph using POS tags for higher accuracy.

🔍 What This File Does

Tokenizes text

Assigns Part-of-Speech (POS) tags

Maps NLTK POS tags to WordNet POS tags

Lemmatizes each word using its correct grammatical role

🧠 Why This Works

Lemmatization without POS tags is inaccurate.
Using POS tags ensures verbs, nouns, adjectives, and adverbs are reduced correctly.

Example:

hanging → hang (verb)

bats → bat (noun)

📚 Sources:

WordNet Lemmatizer: https://www.nltk.org/api/nltk.stem.wordnet.html

POS Tagging (NLTK): https://www.nltk.org/book/ch05.html

Lemmatization theory (Stanford NLP): https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

3️⃣ Named Entity Recognition (NER)

File: ner_exploration.py

✅ Task

Extract PERSON and GPE entities from a corpus of news articles and count the most frequent people.

🔍 What This File Does

Tokenizes text

Applies POS tagging

Uses ne_chunk() to detect named entities

Extracts:

PERSON (people)

GPE (countries, cities, states)

Outputs identified entities

🧠 Why This Works

NER identifies real-world entities in text.
PERSON and GPE are critical for news analysis, information extraction, and knowledge graphs.

📚 Sources:

NLTK Named Entity Recognition: https://www.nltk.org/book/ch07.html

ne_chunk() API: https://www.nltk.org/api/nltk.chunk.html

NER overview (Stanford NLP): https://nlp.stanford.edu/IR-book/html/htmledition/named-entity-recognition-1.html

4️⃣ Sentiment Classifier

File: sentiment.py

✅ Task

Build a classifier for movie reviews using different feature sets (unigrams, bigrams, TF-IDF) and compare accuracies.

🔍 What This File Does

Loads the NLTK Movie Reviews corpus

Extracts word features (unigrams)

Trains a Naive Bayes classifier

Evaluates accuracy on test data

Displays the most informative features

🧠 Why This Works

Naive Bayes is a standard baseline algorithm for sentiment analysis.
Unigram features are simple yet effective for text classification.

📚 Sources:

NLTK Movie Reviews Corpus: https://www.nltk.org/nltk_data/

Naive Bayes Classifier: https://www.nltk.org/api/nltk.classify.html

Text classification theory (Stanford NLP): https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-1.html

📌 Note:
The structure allows easy extension to bigrams and TF-IDF, which are commonly compared in academic experiments.

5️⃣ Topic Word Identification

File: topic_words.py

✅ Task

Use frequency analysis and POS tags to identify candidate topic words (nouns) from an article.

🔍 What This File Does

Tokenizes text

Applies POS tagging

Selects only nouns (NN, NNS, NNP, NNPS)

Performs frequency analysis

Outputs the most frequent nouns as topic candidates

🧠 Why This Works

Nouns usually represent topics, subjects, and key concepts.
Combining POS tagging with frequency analysis is a common lightweight topic-extraction approach.

📚 Sources:

POS Tag Set (Penn Treebank): https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

Topic identification (Stanford NLP): https://nlp.stanford.edu/IR-book/html/htmledition/text-categorization-1.html

Python Counter: https://docs.python.org/3/library/collections.html#collections.Counter

✅ Summary

This repository successfully demonstrates:

✔ Core NLP preprocessing
✔ Linguistic analysis using POS tags
✔ Named Entity Recognition
✔ Machine learning–based sentiment analysis
✔ Topic identification techniques

All tasks are implemented using standard, well-documented NLP methods and are suitable for:

University coursework

GitHub portfolios

NLP learning projects