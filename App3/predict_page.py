import streamlit as st
import numpy as np

# Function to get user inputs
def get_user_input():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income', min_value=0)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
    loan_amount = st.number_input('Loan Amount', min_value=0)
    loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
    credit_history = st.selectbox('Credit History', [1.0, 0.0])
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

    user_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    return user_data

# Function to preprocess the input data
def preprocess_input(data):
    data['Gender'] = 1 if data['Gender'] == 'Male' else 0
    data['Married'] = 1 if data['Married'] == 'Yes' else 0
    data['Dependents'] = 3 if data['Dependents'] == '3+' else int(data['Dependents'])
    data['Education'] = 1 if data['Education'] == 'Graduate' else 0
    data['Self_Employed'] = 1 if data['Self_Employed'] == 'Yes' else 0
    data['Property_Area'] = 0 if data['Property_Area'] == 'Urban' else (1 if data['Property_Area'] == 'Semiurban' else 2)
    
    return np.array(list(data.values())).reshape(1, -1)
