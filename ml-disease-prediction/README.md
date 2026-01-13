# 🏥 Disease Prediction Using Machine Learning  
### Decision Tree (ID3) & Categorical Naive Bayes

This repository demonstrates **disease prediction based on symptoms** using two classical **supervised machine learning algorithms**:

1. **Decision Tree (ID3 algorithm)**
2. **Categorical Naive Bayes**

Both implementations are written in **Python** using **scikit-learn** and are designed to be **beginner-friendly**, **well-commented**, and suitable for **academic and learning purposes**.

---

## 📁 Repository Structure

decision_tree_disease_prediction.py  
naive_bayes_disease_prediction.py  
README.md  

---

## 📄 File Descriptions

### 1️⃣ decision_tree_disease_prediction.py

This file implements **disease prediction using a Decision Tree classifier (ID3)**.

**Algorithm Used**
- Decision Tree Classifier
- Criterion: **Entropy (Information Gain)** → ID3 algorithm

**How It Works**
- Uses historical patient data (symptoms + disease)
- Encodes categorical values into numbers
- Builds a decision tree using entropy
- Learns IF–THEN rules
- Predicts disease based on user-entered symptoms

**Key Concepts**
- Entropy
- Information Gain
- Rule-based classification
- Supervised learning
- Categorical data encoding

**Example Rule Learned**
IF Symptom 1 = Paralysis AND Symptom 2 = Vomiting  
THEN Disease = Ritengitis

---

### 2️⃣ naive_bayes_disease_prediction.py

This file implements **disease prediction using the Categorical Naive Bayes algorithm**.

**Algorithm Used**
- Categorical Naive Bayes
- Based on **Bayes’ Theorem**

**How It Works**
- Calculates probabilities of diseases given symptoms
- Assumes symptoms are conditionally independent
- Chooses the disease with the highest probability
- Uses frequency-based probability estimation

**Core Formula**
P(Disease | Symptoms) =  
(P(Symptoms | Disease) × P(Disease)) / P(Symptoms)

**Key Concepts**
- Probabilistic classification
- Bayes’ theorem
- Feature independence assumption
- Fast and lightweight prediction

---

## 🧠 Comparison Between the Two Models

Feature | Decision Tree (ID3) | Naive Bayes  
------- | ------------------ | ----------  
Approach | Rule-based | Probability-based  
Core Concept | Entropy | Bayes Theorem  
Interpretability | Very High | Medium  
Speed | Moderate | Fast  
Overfitting | Possible | Less likely  
Best For | Rule learning | Small datasets  

---

## 🗂 Dataset Description

Both files use the same medical dataset:

Symptom 1 | Symptom 2 | Disease  
--------- | --------- | --------  
Diarrhea | Fever | Mesiopathy  
Diarrhea | Vomiting | Mesiopathy  
Paralysis | Headache | Mesiopathy  
Paralysis | Vomiting | Ritengitis  
Paralysis | Vomiting | Ritengitis  

- **Features:** Symptom 1, Symptom 2  
- **Target:** Disease  

---

## ⚙️ Data Preprocessing

Since machine learning models cannot process text directly:

- OrdinalEncoder is used to encode symptoms
- LabelEncoder is used to encode disease labels

Example encoding:  
Diarrhea → 0  
Paralysis → 1  
Mesiopathy → 0  
Ritengitis → 1  

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
pip install pandas scikit-learn

### 2️⃣ Run the Decision Tree Model
python decision_tree_disease_prediction.py

### 3️⃣ Run the Naive Bayes Model
python naive_bayes_disease_prediction.py

### 4️⃣ Enter Symptoms When Prompted
Enter Symptom 1: Paralysis  
Enter Symptom 2: Vomiting  

### 5️⃣ View Prediction
Predicted Disease: Ritengitis

---

## 🎯 Learning Outcomes

By studying this repository, you will learn:

- How supervised machine learning works
- Difference between rule-based and probabilistic models
- How to encode categorical data
- How Decision Tree and Naive Bayes algorithms behave
- How to build exam-ready ML projects

---

## 📚 References (Reputable Sources)

Scikit-learn Decision Trees  
https://scikit-learn.org/stable/modules/tree.html  

Scikit-learn Naive Bayes  
https://scikit-learn.org/stable/modules/naive_bayes.html  

Categorical Naive Bayes  
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html  

Entropy & Information Gain  
https://scikit-learn.org/stable/modules/tree.html#classification  

Bayes’ Theorem (Stanford CS229)  
https://cs229.stanford.edu/notes2022fall/cs229-notes2.pdf  

OrdinalEncoder  
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html  

LabelEncoder  
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html  

---

## 👤 Author

Kairo  
Machine Learning & Software Engineering Student
