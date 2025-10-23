import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRFClassifier # Required to load the XGBoost model

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fraud Guard - Transaction Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LOAD MODELS ---
# Use st.cache_resource for efficient loading
@st.cache_resource
def load_model_and_scaler():
    """Loads the saved model and scaler from disk."""
    try:
        model = joblib.load('final_fraud_detection_model.joblib')
        scaler = joblib.load('data_scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler files not found. Make sure they are in the app's root folder.")
        return None, None

model, scaler = load_model_and_scaler()

# The exact 29 feature names in the order your model was trained on
feature_order = [f'V{i}' for i in range(1, 29)] + ['Amount']

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("Fraud Guard 🛡️")
    st.image("image.jpeg") # A generic credit card icon
    st.info(
        "This app uses a pre-trained XGBoost model to predict if a credit "
        "card transaction is fraudulent."
    )
    st.warning(
        "**Disclaimer:** This is a project based on a synthetic dataset. "
        "Do not use for real financial decisions."
    )

# --- 4. MAIN PAGE INTERFACE ---
st.title("Credit Card Transaction Analyzer")
st.markdown("Enter the 29 transaction features below to get a prediction.")

if model and scaler:  # Only run if models loaded
    
    # Use a form to collect all inputs at once
    with st.form(key='transaction_form'):
        st.header("Enter Transaction Details")
        
        # --- Amount Input (Most important, so we put it first) ---
        amount = st.number_input(
            label="Transaction Amount ($)",
            min_value=0.0,
            format="%.2f",
            help="Enter the monetary value of the transaction."
        )
        
        # --- V1-V28 Inputs (Hidden in an expander) ---
        st.markdown("---")
        with st.expander("Enter Advanced Anonymized Features (V1 - V28)"):
            # Create a dictionary to hold the 28 'V' inputs
            v_inputs = {}
            
            # Use columns for a cleaner layout
            cols = st.columns(4)
            
            for i in range(1, 29):
                feature = f'V{i}'
                # Distribute inputs across the 4 columns
                col = cols[(i-1) % 4] 
                v_inputs[feature] = col.number_input(
                    label=feature,
                    format="%.6f",
                    value=0.0,
                    key=feature
                )
        
        # Submit button for the form
        st.markdown("---")
        submit_button = st.form_submit_button(
            label="Analyze Transaction",
            use_container_width=True
        )

    # --- 5. PREDICTION LOGIC (after form is submitted) ---
    if submit_button:
        # 1. Create the full list of inputs in the correct order
        try:
            data_list = [v_inputs[f'V{i}'] for i in range(1, 29)] + [amount]
            
            # 2. Convert to 2D numpy array
            data_array = np.array(data_list).reshape(1, -1)
            
            # 3. Apply the saved RobustScaler
            scaled_data = scaler.transform(data_array)
            
            # 4. Make prediction and get probabilities
            prediction = model.predict(scaled_data)
            prediction_proba = model.predict_proba(scaled_data)
            
            prob_fraud = prediction_proba[0][1]
            prob_not_fraud = prediction_proba[0][0]

            # 6. Display the result
            st.header("Analysis Result")
            if prediction[0] == 1:
                st.error(f"Prediction: FRAUDULENT TRANSACTION 🚨", icon="🚫")
            else:
                st.success(f"Prediction: NOT FRAUDULENT ✅", icon="🛡️")

            # Display probabilities in styled "metric" boxes
            col1, col2 = st.columns(2)
            col1.metric(
                "Fraud Probability",
                f"{prob_fraud:.2%}",
                f"{prob_fraud - prob_not_fraud:.2%} Risk",
                delta_color="inverse" if prob_fraud > 0.5 else "normal"
            )
            col2.metric(
                "Not Fraud Probability",
                f"{prob_not_fraud:.2%}",
                "Safety",
                delta_color="off"
            )
            
            # Show a detailed breakdown
            with st.expander("See Prediction Details"):
                st.write(f"**Class 0 (Not Fraud) Probability:** {prob_not_fraud:.4f}")
                st.write(f"**Class 1 (Fraud) Probability:** {prob_fraud:.4f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.error("Model and scaler could not be loaded. The app cannot function.")