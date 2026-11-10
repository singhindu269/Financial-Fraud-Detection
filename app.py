
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load saved models
@st.cache_resource
def load_models():
    model = joblib.load("Tuned_Random_Forest.pkl")
    scaler = joblib.load("Robust_Scaler.pkl")
    features = joblib.load("Feature_Columns.pkl")
    return model, scaler, features

st.title("💳 Financial Fraud Detection System")
st.markdown(
    "### Machine Learning Based Credit Card Transaction Analysis"
)

st.write(
    "Upload your credit card transaction CSV file "
    "to identify potentially fraudulent transactions."
)

try:
    model, scaler, features = load_models()
    st.success("Machine learning model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.divider()

st.subheader("Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Remove target column if present
        input_data = data.drop(
            columns=["Class"],
            errors="ignore"
        )

        # Check required columns
        missing_columns = [
            col for col in features
            if col not in input_data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        # Maintain training feature order
        input_data = input_data[features].copy()

        if input_data.isnull().any().any():
            st.error("Dataset contains missing values.")
            st.stop()

        # Apply the same scaling used during training
        input_data[["Time", "Amount"]] = scaler.transform(
            input_data[["Time", "Amount"]]
        )

        if st.button("Detect Fraud", type="primary"):

            predictions = model.predict(input_data)

            results = data.copy()

            results["Prediction"] = predictions
            results["Transaction_Status"] = [
                "Fraudulent" if value == 1 else "Genuine"
                for value in predictions
            ]

            fraud_count = int((predictions == 1).sum())
            genuine_count = int((predictions == 0).sum())

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Transactions",
                len(predictions)
            )

            col2.metric(
                "Genuine Transactions",
                genuine_count
            )

            col3.metric(
                "Fraudulent Transactions",
                fraud_count
            )

            st.subheader("Prediction Results")

            st.dataframe(
                results,
                use_container_width=True
            )

            st.subheader("Fraud Detection Summary")

            st.bar_chart(
                pd.Series(
                    {
                        "Genuine": genuine_count,
                        "Fraudulent": fraud_count
                    }
                )
            )

            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Processing error: {e}")

st.divider()

st.caption(
    "Financial Fraud Detection | Machine Learning Project"
)
