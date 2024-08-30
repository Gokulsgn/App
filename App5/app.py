import streamlit as st
import joblib
import os

# Set the path to the model file (update to your actual model path)
model_path = r'C:\Users\gokul\Documents\GitHub\App\App5\trained_model.pkl'

# Display the current working directory for debugging purposes
st.text(f"Current working directory: {os.getcwd()}")

# Load the trained model
def load_model(model_path):
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")
            return None
    else:
        st.error("Model file not found. Please check the path.")
        st.text(f"Checked path: {model_path}")
        return None

# Streamlit app
def main():
    st.title("Track Popularity Prediction")

    # Load the model
    model = load_model(model_path)
    
    if model is None:
        return  # Stop the app if the model couldn't be loaded

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
        try:
            prediction = model.predict(input_data_as_numpy_array)
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

if __name__ == "__main__":
    main()
