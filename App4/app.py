import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained machine learning model
def load_model():
    with open(r'C:\Users\gokul\Documents\GitHub\App\App4\logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Function to get user input
def get_user_input():
    st.sidebar.header("User Input")

    # Define input fields
    age = st.sidebar.slider("Age", 0, 120, 50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 0, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 0, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 0, 250, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("Depression Induced by Exercise Relative to Rest", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

    # Create a DataFrame from the user input
    user_input = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "Male" else 0],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    return user_input

def main():
    st.title("Heart Disease Prediction App")

    # Get user input
    user_data = get_user_input()

    # Display user input
    st.write("### User Input:")
    st.write(user_data)

    # Make prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(user_data)[0]
            prediction_proba = model.predict_proba(user_data)[0]

            # Display prediction result
            st.write(f"### Prediction: {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'}")

            # Plot prediction probabilities
            st.write("### Prediction Probabilities:")
            fig, ax = plt.subplots()
            ax.bar(['No Heart Disease', 'Heart Disease'], prediction_proba)
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
