import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# Function to preprocess and clean the dataset
def preprocess_data(data):
    # List of columns to drop (remove if they exist in the dataset)
    columns_to_drop = ['...1', 'ID']
    
    # Drop only columns that exist in the dataset
    data_cleaned = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # Fill missing values with mean for numeric columns
    data_cleaned.fillna(data_cleaned.mean(), inplace=True)
    
    # Encode categorical column 'COPDSEVERITY'
    data_cleaned['COPDSEVERITY'] = data_cleaned['COPDSEVERITY'].map({
        'MILD': 1,
        'MODERATE': 2,
        'SEVERE': 3,
        'VERY SEVERE': 4
    })
    
    # Separate features (X) and target (y)
    X = data_cleaned.drop(columns=['copd'])
    y = data_cleaned['copd']
    return X, y, data_cleaned

# Function to predict COPD based on symptoms
def predict_copd(symptoms, mrc_grade, spirometry_grade):
    risk_score = sum(symptoms) + mrc_grade + spirometry_grade
    if risk_score >= 8:
        return "High Risk: COPD Likely"
    elif risk_score >= 5:
        return "Moderate Risk: Further Evaluation Needed"
    else:
        return "Low Risk: No COPD Detected"

# Function to classify severity based on MRC grade
def mrc_classification(grade):
    if grade == 1:
        return "Mild: Only with heavy exercise"
    elif grade == 2:
        return "Moderate: Shortness of breath when walking fast or uphill"
    elif grade == 3:
        return "Severe: Slower walking and stopping frequently to breathe"
    elif grade == 4:
        return "Very Severe: Stopping after walking 90 meters"
    elif grade == 5:
        return "Extremely Severe: Shortness of breath even when dressing"

# Function to classify spirometry severity
def spirometry_classification(fev1_percentage):
    if fev1_percentage >= 80:
        return "Mild"
    elif 50 <= fev1_percentage < 80:
        return "Moderate"
    elif 30 <= fev1_percentage < 50:
        return "Severe"
    else:
        return "Very Severe"

# Load the dataset
dataset_path = 'Cleaned_COPD_Data.csv'  # Path to your dataset
data = pd.read_csv(dataset_path)

# Preprocess the data
X, y, data_cleaned = preprocess_data(data)

# Train the Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the trained model (use a local path for file saving)
model_filename = 'copd_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(rf, file)

# Evaluate the model
accuracy = rf.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Display cleaned data for review
st.subheader("Cleaned COPD Data")
st.write(data_cleaned)

# Input questions for respiratory symptoms
st.title('COPD Detection and Management System')

st.header('Enter Patient Symptoms and Information')
age = st.number_input('Age:', min_value=0, max_value=100, value=60)
shortness_of_breath = st.radio('Shortness of breath at rest or exertion?', ['Yes', 'No'])
activity_limit = st.radio('Activity limitation?', ['Yes', 'No'])
cough = st.radio('Cough?', ['Yes', 'No'])
sputum_production = st.radio('Sputum production?', ['Yes', 'No'])
wheezing = st.radio('Wheezing?', ['Yes', 'No'])
smoker = st.radio('Current or past smoker?', ['Yes', 'No'])
respiratory_infections = st.radio('Frequent respiratory infections?', ['Yes', 'No'])

# Convert symptoms into numeric values for AI/ML model
symptoms = [
    1 if shortness_of_breath == 'Yes' else 0,
    1 if activity_limit == 'Yes' else 0,
    1 if cough == 'Yes' else 0,
    1 if sputum_production == 'Yes' else 0,
    1 if wheezing == 'Yes' else 0,
    1 if smoker == 'Yes' else 0,
    1 if respiratory_infections == 'Yes' else 0
]

# MRC grading for dyspnea
mrc_grade = st.slider('Grade of Dyspnea (MRC scale):', 1, 5, value=1)

# Spirometry test input
fev1_percentage = st.number_input('Enter FEV1 as percentage of predicted value:', 0, 100, value=80)

# Classify MRC and Spirometry
mrc_severity = mrc_classification(mrc_grade)
spirometry_severity = spirometry_classification(fev1_percentage)

# Show severity classifications
st.write(f"MRC Classification: {mrc_severity}")
st.write(f"Spirometry Classification: {spirometry_severity}")

# Button for prediction
if st.button('Predict COPD'):
    diagnosis = predict_copd(symptoms, mrc_grade, fev1_percentage)
    st.write(diagnosis)

    # Management recommendations based on risk
    st.header('Management Recommendations')

    if diagnosis == "High Risk: COPD Likely":
        st.write("1. Refer to a pulmonologist for further evaluation.")
        st.write("2. Consider spirometry test for definitive diagnosis.")
        st.write("3. Start pharmacological treatment (e.g., corticosteroids, inhalers).")
        st.write("4. Encourage smoking cessation and regular physical activity.")
        st.write("5. Consider long-term oxygen therapy if SpO2 < 90% at rest.")

    elif diagnosis == "Moderate Risk: Further Evaluation Needed":
        st.write("1. Monitor symptoms and repeat screening if necessary.")
        st.write("2. Encourage lifestyle changes: stop smoking, increase physical activity.")
        st.write("3. Vaccination for influenza and pneumonia.")
        
    else:
        st.write("1. Encourage healthy lifestyle choices.")
        st.write("2. Regular monitoring and follow-up if necessary.")

    # Non-pharmacologic recommendations
    st.subheader("Non-Pharmacologic Management:")
    st.write("1. **Education**: Teach patient about COPD and self-management strategies.")
    st.write("2. **Smoking Cessation**: Essential to slowing disease progression.")
    st.write("3. **Avoidance of Risk Factors**: Reduce exposure to irritants (e.g., air pollution).")
    st.write("4. **Energy Conservation**: Manage daily activities to avoid excessive fatigue.")
    st.write("5. **Palliative Care**: For progressive disease, provide support for patient and family.")
    
    # Pharmacologic management recommendations
    st.subheader("Pharmacologic Management:")
    st.write("1. **Bronchodilators**: Primary treatment for relieving shortness of breath.")
    st.write("2. **Combination ICS and LABA**: For moderate to severe COPD, reduces inflammation.")
    st.write("3. **Oxygen Therapy**: For chronic hypoxemia, improves life expectancy and quality of life.")
    st.write("4. **Vaccination**: Influenza and pneumococcal vaccines to prevent infections.")
    st.write("5. **Antibiotics**: Prescribed if bacterial infection is suspected during acute exacerbation.")

    # Acute exacerbation management
    st.subheader("Acute Exacerbation Management:")
    st.write("1. **Oxygen Therapy**: To increase oxygen levels in the blood.")
    st.write("2. **Inhaled Bronchodilators**: To relieve bronchospasm.")
    st.write("3. **Systemic Corticosteroids**: To reduce inflammation.")
    st.write("4. **Antibiotics**: If bacterial infection is suspected.")
    
   # Referral to Specialist
    st.subheader("Referral to Specialist:")
    st.write("1. If unsure about diagnosis or if symptoms don't match level of airway obstruction.")
    st.write("2. If rapid decline in lung function occurs (e.g., FEV1 decreases by >80 ml/year).")
    st.write("3. If suspected alpha-1-antitrypsin deficiency.")
    st.write("4. If no response to therapy or if acute exacerbations are severe or frequent.")
