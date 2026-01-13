# ============================================================
# DISEASE PREDICTION USING DECISION TREE (ID3)
# ============================================================
# This program predicts a disease based on two symptoms
# using a Decision Tree classifier (ID3 algorithm).
#
# MAIN CONCEPTS USED:
# - Categorical data encoding
# - Decision Tree (Entropy / Information Gain)
# - Supervised Machine Learning
# - User input prediction
# ============================================================


# ------------------------------------------------------------
# STEP 0: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

# pandas:
# Used to create and manipulate tabular data (like Excel tables)
import pandas as pd

# DecisionTreeClassifier:
# Machine Learning model that learns rules in a tree structure
from sklearn.tree import DecisionTreeClassifier

# OrdinalEncoder:
# Converts categorical feature values (text) into numbers
# Example: "Diarrhea" → 0, "Paralysis" → 1
#
# LabelEncoder:
# Converts target labels (diseases) into numeric form
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


# ------------------------------------------------------------
# STEP 1: CREATE THE MEDICAL DATASET
# ------------------------------------------------------------
# This dataset represents HISTORICAL patient records.
# Each row = one patient
# Columns:
# - Symptom 1
# - Symptom 2
# - Disease (TARGET / CLASS LABEL)

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

# At this stage:
# The model CANNOT work with text values
# We must convert everything into numbers


# ------------------------------------------------------------
# STEP 2: DATA PREPROCESSING (ENCODING)
# ------------------------------------------------------------

# Create encoder objects
encoder = OrdinalEncoder()
label_encoder = LabelEncoder()

# FEATURES = INPUT VARIABLES
# These are used to MAKE predictions
features = ['Symptom 1', 'Symptom 2']

# X = feature data (independent variables)
X = data[features]

# Convert text symptoms into numeric values
# Example:
# Diarrhea → 0
# Paralysis → 1
# Fever → 0
# Vomiting → 1
X_encoded = encoder.fit_transform(X)

# y = target variable (dependent variable)
# This is what we want to predict
y = data['Disease']

# Convert disease names into numbers
# Example:
# Mesiopathy → 0
# Ritengitis → 1
y_encoded = label_encoder.fit_transform(y)


# ------------------------------------------------------------
# STEP 3: TRAIN THE DECISION TREE MODEL (ID3)
# ------------------------------------------------------------

# Create Decision Tree model
# criterion='entropy' means:
# - Use ENTROPY to measure impurity
# - Choose splits using INFORMATION GAIN
# This is exactly how ID3 works
model = DecisionTreeClassifier(criterion='entropy')

# Train the model:
# The tree learns rules such as:
# IF Symptom 1 = Paralysis AND Symptom 2 = Vomiting
# THEN Disease = Ritengitis
model.fit(X_encoded, y_encoded)


# ------------------------------------------------------------
# STEP 4: USER INPUT FOR PREDICTION
# ------------------------------------------------------------

print("\n--- INPUT FOR DISEASE PREDICTION ---")

# Show valid options so user does not enter unknown symptoms
print("Available Symptom 1 options:", data['Symptom 1'].unique())
print("Available Symptom 2 options:", data['Symptom 2'].unique())

try:
    # Take symptom input from the user
    symptom1_input = input("Enter Symptom 1 (e.g., Paralysis): ").strip()
    symptom2_input = input("Enter Symptom 2 (e.g., Fever): ").strip()

    # Convert user input into DataFrame
    # (Must match training data structure)
    new_patient_data = pd.DataFrame({
        'Symptom 1': [symptom1_input],
        'Symptom 2': [symptom2_input]
    })

    # Encode the new patient symptoms
    # Uses SAME encoder learned during training
    X_new_encoded = encoder.transform(new_patient_data[features])


    # --------------------------------------------------------
    # STEP 5: PREDICTION
    # --------------------------------------------------------

    # Predict disease (numeric output)
    predicted_encoded = model.predict(X_new_encoded)[0]

    # Convert numeric prediction back to disease name
    predicted_disease = label_encoder.inverse_transform(
        [predicted_encoded]
    )[0]

    print("\n--- DECISION TREE (ID3) PREDICTION RESULT ---")
    print(f"Symptoms Entered: {symptom1_input}, {symptom2_input}")
    print(f"Predicted Disease: {predicted_disease}")

except ValueError as e:
    # This happens if user enters a symptom
    # that was NOT present in training data
    print("\nERROR: Unknown symptom entered.")
    print("Please choose from the available symptom options.")
    print("Technical details:", e)
