import streamlit as st
import numpy as np
import pickle
import os

# Load the pre-trained machine learning model
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
        st.write(f"Loading model from: {model_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            st.write("Model successfully loaded.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
    return model

model = load_model()

# Streamlit app
def main():
    st.title("Track Popularity Prediction")

    # Sidebar input fields
    st.sidebar.header("User Input")
    id = st.sidebar.text_input("ID", "1")
    name = st.sidebar.text_input("Name", "Track1")
    genre = st.sidebar.selectbox("Genre", ["Pop", "Rock", "Jazz", "Classical", "Hip-Hop"])
    artists = st.sidebar.text_input("Artists", "Artist1")
    album = st.sidebar.text_input("Album", "Album1")
    popularity = st.sidebar.slider("Popularity", 0, 100, 50)
    duration_ms = st.sidebar.number_input("Duration (ms)", min_value=0, value=210000)
    explicit = st.sidebar.selectbox("Explicit", ["No", "Yes"])
    duration_min = st.sidebar.number_input("Duration (min)", min_value=0.0, value=3.5)

    # Convert inputs to numerical format
    genre_encoded = {"Pop": 0, "Rock": 1, "Jazz": 2, "Classical": 3, "Hip-Hop": 4}[genre]
    explicit_encoded = 1 if explicit == "Yes" else 0

    # Create feature array
    input_data = [popularity, duration_ms, genre_encoded, explicit_encoded, duration_min]
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Prediction button
    if st.button("Predict"):
        if model is not None:
            try:
                prediction = model.predict(input_data_as_numpy_array)
                if hasattr(model, "predict_proba"):
                    prediction_proba = model.predict_proba(input_data_as_numpy_array)
                    prediction_text = f'Popular (Confidence: {prediction_proba[0][1]:.2f})' if prediction[0] > 0.5 else f'Not Popular (Confidence: {prediction_proba[0][0]:.2f})'
                else:
                    prediction_text = 'Popular' if prediction[0] > 0.5 else 'Not Popular'
                
                st.markdown(
                    f"""
                    <div class="main-content">
                        <h2 style='font-size:40px;'>Prediction: {prediction_text}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.error("Model is not loaded properly.")

if __name__ == "__main__":
    main()
