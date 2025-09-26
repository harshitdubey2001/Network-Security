import streamlit as st
import pandas as pd
import pickle

# Load the trained model (from .pkl file)
@st.cache_resource
def load_model():
    with open("final_model/model.pkl", "rb") as f:   # adjust path if needed
        return pickle.load(f)

model = load_model()

st.title("Network Security - Phishing Detection")
st.write("Upload data to check if it's **Phishing** or **Legit**")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Drop target column if present
    if "Result" in df.columns:
        X = df.drop(columns=["Result"])
    else:
        X = df

    # Predictions
    preds = model.predict(X)
    df["Prediction"] = ["Phishing" if p == 1 else "Legit" for p in preds]

    st.subheader("Prediction Results")
    st.dataframe(df.head())

    # Download results
    st.download_button(
        label="Download Predictions as CSV",
        data=df.to_csv(index=False),
        file_name="phishing_predictions.csv",
        mime="text/csv",
    )

# Manual single entry check (works only if your model was trained on numeric features)
st.subheader("Check Single Entry")
st.write("Enter feature values as comma-separated input (matching your dataset columns).")

user_input = st.text_input("Example: 1,0,1,0,0,1 ...")

if st.button("Predict Single Entry"):
    if user_input.strip() != "":
        try:
            values = [float(x) for x in user_input.split(",")]
            single_df = pd.DataFrame([values], columns=X.columns)  # match training columns
            pred = model.predict(single_df)[0]
            st.write("Prediction:", "Phishing" if pred == 1 else "Legit")
        except Exception as e:
            st.error(f"Error: {e}")
