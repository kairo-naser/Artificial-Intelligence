# ============================================================
# DISEASE PREDICTION USING CATEGORICAL NAIVE BAYES
# ============================================================
# This program predicts a disease based on symptoms
# using the CATEGORICAL NAIVE BAYES algorithm.
#
# MAIN CONCEPTS USED:
# - Supervised Machine Learning
# - Probabilistic classification
# - Bayes Theorem
# - Categorical data encoding
# - User-based prediction
# ============================================================


# ------------------------------------------------------------
# STEP 0: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

# pandas:
# Used for creating and handling tabular datasets
import pandas as pd

# CategoricalNB:
# Naive Bayes classifier designed for categorical features
from sklearn.naive_bayes import CategoricalNB

# accuracy_score:
# Used to evaluate model accuracy (not used here, but useful)
from sklearn.metrics import accuracy_score

# OrdinalEncoder:
# Converts categorical feature values into numeric form
#
# LabelEncoder:
# Converts target class labels (Disease names) into numbers
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


# ------------------------------------------------------------
# STEP 1: CREATE THE MEDICAL DATASET
# ------------------------------------------------------------
# This dataset represents historical medical records.
#
# Each row = one patient
# Columns:
# - Symptom 1
# - Symptom 2
# - Disease (TARGET CLASS)

data = pd.DataFrame({
    'Symptom 1': [
        'Diarrhea',
        'Diarrhea',
        'Paralysis',
        'Paralysis',
        'Paralysis'
    ],
    'Symptom 2': [
        'Fever',
        'Vomiting',
        'Headache',
        'Vomiting',
        'Vomiting'
    ],
    'Disease': [
        'Mesiopathy',
        'Mesiopathy',
        'Mesiopathy',
        'Ritengitis',
        'Ritengitis'
    ]
})

# IMPORTANT:
# Naive Bayes requires numeric values
# So text data must be encoded before training


# ------------------------------------------------------------
# STEP 2: DATA PREPROCESSING (ENCODING)
# ------------------------------------------------------------

# Create encoder objects
encoder = OrdinalEncoder()
label_encoder = LabelEncoder()

# FEATURES (INPUT VARIABLES)
# These are the predictors
features = ['Symptom 1', 'Symptom 2']

# X = feature data
X = data[features]

# Convert categorical symptom values into numbers
# Example:
# Diarrhea → 0
# Paralysis → 1
# Fever → 0
# Vomiting → 1
X_encoded = encoder.fit_transform(X)

# TARGET VARIABLE
# This is what the model learns to predict
y = data['Disease']

# Encode disease labels into numbers
# Example:
# Mesiopathy → 0
# Ritengitis → 1
y_encoded = label_encoder.fit_transform(y)


# ------------------------------------------------------------
# STEP 3: TRAIN THE CATEGORICAL NAIVE BAYES MODEL
# ------------------------------------------------------------

# Create the Naive Bayes model
#
# Naive Bayes assumes:
# 1. Features are independent
# 2. Uses Bayes Theorem:
#
# P(Disease | Symptoms) =
# (P(Symptoms | Disease) * P(Disease)) / P(Symptoms)
#
# The model calculates probabilities from training data
model = CategoricalNB()

# Train the model using encoded data
model.fit(X_encoded, y_encoded)


# ------------------------------------------------------------
# STEP 4: USER INPUT FOR PREDICTION
# ------------------------------------------------------------

print("\n--- INPUT FOR DISEASE PREDICTION ---")

# Display valid symptom options
print("Available Symptom 1 options:", data['Symptom 1'].unique())
print("Available Symptom 2 options:", data['Symptom 2'].unique())

try:
    # Take symptom input from the user
    symptom1_input = input("Enter Symptom 1 (e.g., Paralysis): ").strip()
    symptom2_input = input("Enter Symptom 2 (e.g., Fever): ").strip()

    # Create a DataFrame for the new patient
    # Must match the structure of training data
    new_patient_data = pd.DataFrame({
        'Symptom 1': [symptom1_input],
        'Symptom 2': [symptom2_input]
    })

    # Encode user input
    # Uses SAME encoding rules learned earlier
    X_new_encoded = encoder.transform(new_patient_data[features])


    # --------------------------------------------------------
    # STEP 5: MAKE PREDICTION
    # --------------------------------------------------------

    # Predict disease (numeric output)
    predicted_result_encoded = model.predict(X_new_encoded)[0]

    # Convert numeric prediction back to disease name
    predicted_disease = label_encoder.inverse_transform(
        [predicted_result_encoded]
    )[0]

    print("\n--- NAIVE BAYES PREDICTION RESULT ---")
    print(f"Symptoms Entered: Symptom 1 = {symptom1_input}, Symptom 2 = {symptom2_input}")
    print(f"Predicted Disease: {predicted_disease}")

except ValueError as e:
    # This error occurs if user enters
    # a symptom not seen during training
    print("\nERROR: Unknown symptom entered.")
    print("Please choose symptoms from the available options.")
    print("Technical details:", e)
