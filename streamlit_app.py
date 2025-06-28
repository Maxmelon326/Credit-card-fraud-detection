# Updated app.py with preprocessing for both new and existing users

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

# Load models
new_user_model = joblib.load("new_user_model")
existing_user_model = joblib.load("existing_user_model")

# Load dataset for existing user lookup
train_data = pd.read_csv("train.csv")  # Ensure this is a small sample version if on Streamlit Cloud

# Preprocessing for new user
occupation_mapping = {'Laborers': 0, 'Core staff': 1, 'Managers': 2, 'Other': 3}
def preprocess_new_user(df):
    df['log_income'] = np.log1p(df['net_yearly_income'])
    df['log_days_employed'] = np.log1p(df['no_of_days_employed'])
    df['income_per_member'] = df['net_yearly_income'] / (df['total_family_members'] + 1)
    df['is_home_owner'] = df['owns_house']
    df['occupation_type'] = df['occupation_type'].map(occupation_mapping).fillna(3)  # Map occupation
    df = df[['age', 'log_income', 'log_days_employed', 'income_per_member', 'is_home_owner', 'occupation_type']]
    return df

# Preprocessing for existing user
exist_cols = ['age', 'credit_score', 'credit_limit', 'credit_limit_used(%)',
              'yearly_debt_payments', 'prev_defaults', 'default_in_last_6months']
def preprocess_existing_user(df):
    imputer = SimpleImputer(strategy='median')
    df = pd.DataFrame(imputer.fit_transform(df), columns=exist_cols)
    return df

# --- Streamlit App ---
st.title("🏦 FinTech Credit Risk Assessment Demo")

# Step 1: Select user type
user_type = st.radio("Are you a new or existing user?", ["New User", "Existing User"])

# Step 2A: New user inputs
if user_type == "New User":
    st.subheader("📝 Please enter your personal information")
    age = st.slider("Age", 18, 100, 30)
    net_income = st.number_input("Net Yearly Income", min_value=0.0)
    days_employed = st.number_input("No. of Days Employed", min_value=0.0)
    total_family = st.slider("Family Members", 0, 10, 1)
    owns_house = st.selectbox("Owns House?", ["Yes", "No"])
    occupation_type = st.selectbox("Occupation Type", list(occupation_mapping.keys()))

    user_input = pd.DataFrame({
        "age": [age],
        "net_yearly_income": [net_income],
        "no_of_days_employed": [days_employed],
        "total_family_members": [total_family],
        "owns_house": [1 if owns_house == "Yes" else 0],
        "occupation_type": [occupation_type]
    })

    if st.button("Evaluate Credit"):
        processed_input = preprocess_new_user(user_input)
        prob = new_user_model.predict_proba(processed_input)[0][1]
        st.metric("Predicted Default Probability", f"{prob:.2%}")
        if prob < 0.1: # You can set your rejection rate here.
            st.success("✅ Credit Approved")
        else:
            st.error("❌ Credit Rejected")

# Step 2B: Existing user by ID
elif user_type == "Existing User":
    st.subheader("🔍 Enter your Customer ID")
    input_id = st.text_input("Customer ID (e.g., CST_115179)")

    if st.button("Search and Evaluate"):
        if input_id in train_data["customer_id"].values:
            user_row = train_data[train_data["customer_id"] == input_id]
            X_existing = user_row[exist_cols]
            processed_exist = preprocess_existing_user(X_existing)
            prob = existing_user_model.predict_proba(processed_exist)[0][1]
            st.metric("Predicted Default Probability", f"{prob:.2%}")
            if prob < 0.2:
                st.success("✅ Credit Approved")
            else:
                st.error("❌ Credit Rejected")
        else:
            st.warning("Customer ID not found.")
