# 😊 Sentiment Analysis Using NLTK (VADER)

This project demonstrates **sentiment analysis** using the **VADER Sentiment Analyzer** from the **Natural Language Toolkit (NLTK)**.

The program classifies a given sentence as:
- Positive
- Negative
- Neutral

It is designed for **learning Natural Language Processing (NLP)** and is suitable for **academic and beginner-level projects**.

---

## 📁 Project Structure

sentiment-analysis-nltk/  
│── sentiment_analysis.py  
│── README.md  

---

## 📄 File Description

### `sentiment_analysis.py`

This Python file performs **sentiment analysis** using a rule-based NLP approach.

**Key Features**
- Uses NLTK’s VADER lexicon
- Works well on short sentences
- No training data required
- Fast and lightweight

---

## 🧠 Algorithm Used: VADER

**VADER (Valence Aware Dictionary and sEntiment Reasoner)** is a:
- Rule-based sentiment analysis tool
- Designed for social media and short texts
- Lexicon-based (uses predefined word scores)

---

## 📊 Sentiment Scoring Method

VADER produces four scores:
- `positive`
- `negative`
- `neutral`
- `compound` (final sentiment score)

**Compound score interpretation:**
- `>= 0.05` → Positive
- `<= -0.05` → Negative
- Otherwise → Neutral

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Library
```bash
pip install nltk
