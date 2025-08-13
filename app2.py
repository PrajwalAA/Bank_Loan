import streamlit as st
import pandas as pd
import joblib # For loading the trained model

# Suppress warnings for cleaner output in Streamlit
import warnings
warnings.simplefilter('ignore')

# --- 1. Model Loading ---
# Ensure 'bank_loan_model2.pkl' is in the same directory as this Streamlit app.py
MODEL_PATH = 'bank_loan_model2.pkl'

try:
    loaded_model = joblib.load(MODEL_PATH)
    st.success("Bank loan prediction model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file not found. Please ensure '{MODEL_PATH}' is in the correct directory.")
    st.stop() # Stop the app if the essential file is missing
except Exception as e:
    st.error(f"An error occurred while loading the model file: {e}")
    st.stop()

# --- 2. Streamlit Application Layout ---
st.set_page_config(
    page_title="Personal Loan Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🏦 Personal Loan Eligibility Predictor")
st.markdown("Enter your details to find out your personal loan eligibility.")

# Input widgets for loan features
st.subheader("Your Financial and Personal Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=35, step=1)
    experience = st.number_input("Experience (years)", min_value=0, max_value=60, value=10, step=1)
    income = st.number_input("Monthly Income (in thousands ₹)", min_value=0, max_value=500, value=50, step=5)
    family_size = st.number_input("Family Size (1 to 4)", min_value=1, max_value=4, value=2, step=1)
    credit_spend = st.number_input("Avg Credit Card Spend per month (in thousands ₹)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)

with col2:
    # Map education level to numerical values as used in training (1, 2, 3)
    education_options = {
        "1: Undergraduate": 1,
        "2: Graduate": 2,
        "3: Advanced/Professional": 3
    }
    selected_education_text = st.selectbox("Education Level", list(education_options.keys()))
    education = education_options[selected_education_text]

    mortgage = st.number_input("Mortgage (in thousands ₹)", min_value=0, max_value=1000, value=0, step=10)
    securities = st.radio("Do you have a Securities Account?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    certificate = st.radio("Do you have a Certificate of Deposit (CD) Account?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    online = st.radio("Do you use Online Banking?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    credit_card = st.radio("Do you have a Credit Card?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Button to trigger prediction
if st.button("Check Eligibility"):
    if loaded_model:
        # Create input DataFrame from user data
        # IMPORTANT: Ensure these keys match the exact feature names used during model training
        user_data = {
            "Age": age,
            "Experience": experience,
            "Income": income,
            "Family": family_size,
            "CCAvg": credit_spend,
            "Education": education,
            "Mortgage": mortgage,
            "Securities Account": securities, # Corrected key
            "CD Account": certificate,     # Corrected key
            "Online": online,
            "CreditCard": credit_card
        }
        input_df = pd.DataFrame([user_data])

        try:
            # Make prediction
            prediction = loaded_model.predict(input_df)[0] # Get the single prediction value

            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success("✅ Congratulations! You are likely **Eligible** for a Personal Loan.")
            else:
                st.error("❌ Sorry, you are currently **Not Eligible** for a Personal Loan.")
                st.markdown("---")
                st.subheader("Potential Reasons for Rejection:")
                reasons = []

                # Add specific thresholds based on the analysis provided in your original notebook
                # These thresholds are illustrative and should match your model's decision logic
                if income < 50: # Adjust this threshold based on your model's training data insights
                    reasons.append("Monthly income might be below the typical threshold for loan approval.")
                if credit_spend < 1.0: # Adjust
                    reasons.append("Average monthly credit card spend might be too low, indicating limited credit activity.")
                if education == 1:
                    reasons.append("Applicants with an undergraduate education sometimes face higher scrutiny.")
                if securities == 0:
                    reasons.append("Lack of a securities account could be a factor.")
                if certificate == 0:
                    reasons.append("Absence of a Certificate of Deposit (CD) account might affect eligibility.")
                if online == 0:
                    reasons.append("Not using online banking services could be a minor concern.")
                if credit_card == 0:
                    reasons.append("Lack of a credit card might indicate limited credit history.")
                if experience < 5: # Adjust
                    reasons.append("Insufficient work experience could be a contributing factor.")
                if mortgage == 0 and income > 100: # Example: If high income but no mortgage, might be an anomaly or less need
                    pass # This is a place holder, as a low mortgage is not a reason for rejection.
                if mortgage > 500 and income < 100:
                    reasons.append("High mortgage burden relative to income could be a concern.")


                if reasons:
                    for r in reasons:
                        st.warning(f"- {r}")
                    st.info("These are general indicators. For precise reasons, please consult with a bank representative.")
                else:
                    st.info("Based on your inputs, the model predicts non-eligibility. "
                            "However, specific reasons couldn't be pinpointed from general thresholds. "
                            "Factors like overall financial health, debt-to-income ratio, or specific bank policies might be at play.")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check your input values and ensure the model file is correct.")
    else:
        st.warning("Model not loaded. Please check the file path and restart the app.")

# --- How to Run Instructions ---
st.markdown(
    """
    ---
    ### How to Run This Application:
    1.  **Save** the code above as a Python file (e.g., `loan_app.py`).
    2.  **Ensure** you have trained your model and saved it as `bank_loan_model2.pkl`. The provided notebook code saves it as `model.pkl`, so you might need to rename it or adjust `MODEL_PATH` in this Streamlit app to `"model.pkl"`.
    3.  **Place** the `bank_loan_model2.pkl` (or `model.pkl`) file in the **same directory** as `loan_app.py`.
    4.  **Install** the necessary libraries:
        ```bash
        pip install streamlit pandas scikit-learn joblib
        ```
        *(Note: If your model was trained with `lightgbm`, ensure `pip install lightgbm` is also done, although `joblib` handles the loading.)*
    5.  **Run** the app from your terminal:
        ```bash
        streamlit run loan_app.py
        ```
    6.  Your browser will automatically open to the Streamlit app!
    """
)
