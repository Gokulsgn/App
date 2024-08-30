import streamlit as st
import pickle
from predict_page import preprocess_input, get_user_input

st.set_page_config(
    page_title="Loan Status Prediction",  # Title of the page
    page_icon=":bank:",  # You can use emojis like ":bank:" or a custom favicon
    layout="centered"  # Other options are "wide"
)

# Display the title of the app with the 'title' class to center it
st.markdown("<h1 class='title'>Loan Status Prediction</h1>", unsafe_allow_html=True)




# Load the pre-trained machine learning model
def load_model():
    with open('loan_LR_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

def main():
    # Inject custom CSS to change the background color and header styling
    st.markdown(
        """
        <style>
        body, .stApp, header, .css-1d391kg, .css-18e3th9, .css-1v3fvcr {
            background-color: #08a9b8;
        }
        h1 {
            text-align: center;
            color: white;
        }
        h2, h3, h4, h5, h6 {
            text-align: center;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    # Get user input using the function from predict_page.py
    user_data = get_user_input()

    # Preprocess the user input data
    input_data = preprocess_input(user_data)

    # Make prediction using the model
    prediction = model.predict(input_data)[0]

    # Map the prediction to a readable format
    loan_status = 'Approved' if prediction == 1 else 'Rejected'

    # Display the prediction
    st.write(f'### Loan Status Prediction: **{loan_status}**')

if __name__ == '__main__':
    main()
