import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

# Load models
new_user_model = joblib.load("new_user_model")
existing_user_model = joblib.load("existing_user_model")

# Load dataset to identify existing users
train_data = pd.read_csv("train.csv")

st.title("🏦 FinTech Credit Risk Assessment Demo")

# Step 1: Select user type
user_type = st.radio("Are you a new or existing user?", ["New User", "Existing User"])

# ===== New User Flow =====
if user_type == "New User":
    st.subheader("📝 Please enter your personal information")

    age = st.slider("Age", 18, 100, 30)
    net_income = st.number_input("Net Yearly Income", min_value=0.0)
    days_employed = st.number_input("No. of Days Employed", min_value=0.0)
    total_family = st.slider("Family Members", 0, 10, 1)
    owns_house = st.selectbox("Owns House?", ["Yes", "No"])
    occupation_type = st.selectbox("Occupation Type", [
        "Accountants", "Cleaning staff", "Cooking staff", "Core staff", "Drivers",
        "High skill tech staff", "HR staff", "IT staff", "Laborers", "Low-skill Laborers",
        "Managers", "Medicine staff", "Private service staff", "Realty agents", "Sales staff",
        "Secretaries", "Security staff", "Unknown", "Waiters/barmen staff"
    ])

    user_input = pd.DataFrame({
        "age": [age],
        "net_yearly_income": [net_income],
        "no_of_days_employed": [days_employed],
        "total_family_members": [total_family],
        "owns_house": [1 if owns_house == "Yes" else 0],
        "occupation_type": [occupation_type]
    })

    def preprocess_new_user(df):
        df = df.copy()
        df['log_income'] = np.log1p(df['net_yearly_income'])
        df['log_days_employed'] = np.log1p(df['no_of_days_employed'])
        df['income_per_member'] = df['net_yearly_income'] / (df['total_family_members'] + 1)
        df['is_home_owner'] = df['owns_house']
        df = df.drop(columns=['owns_house'])
        df = pd.get_dummies(df, columns=['occupation_type'])

        occupation_dummies = [
            'occupation_type_Accountants', 'occupation_type_Cleaning staff', 'occupation_type_Cooking staff',
            'occupation_type_Core staff', 'occupation_type_Drivers', 'occupation_type_High skill tech staff',
            'occupation_type_HR staff', 'occupation_type_IT staff', 'occupation_type_Laborers',
            'occupation_type_Low-skill Laborers', 'occupation_type_Managers', 'occupation_type_Medicine staff',
            'occupation_type_Private service staff', 'occupation_type_Realty agents', 'occupation_type_Sales staff',
            'occupation_type_Secretaries', 'occupation_type_Security staff', 'occupation_type_Unknown',
            'occupation_type_Waiters/barmen staff'
        ]

        for col in occupation_dummies:
            if col not in df.columns:
                df[col] = 0

        final_cols = ['age', 'net_yearly_income', 'no_of_days_employed', 'total_family_members',
                      'log_income', 'log_days_employed', 'income_per_member', 'is_home_owner'] + occupation_dummies
        df = df[final_cols]
        return df

    if st.button("Evaluate Credit"):
        processed_input = preprocess_new_user(user_input)
        prob = new_user_model.predict_proba(processed_input)[0][1]
        st.metric("Predicted Default Probability", f"{prob:.2%}")
        if prob < 0.1: # You can set your rejection rate here.
            st.success("✅ Credit Approved")
        else:
            st.error("❌ Credit Rejected")

# ===== Existing User Flow =====
elif user_type == "Existing User":
    st.subheader("🔍 Enter your Customer ID")
    input_id = st.text_input("Customer ID (e.g., CST_115179)")

    if st.button("Search and Evaluate"):
        if input_id in train_data["customer_id"].values:
            user_row = train_data[train_data["customer_id"] == input_id]

            features = [
                'age', 'net_yearly_income', 'occupation_type', 'no_of_days_employed',
                'owns_house', 'total_family_members', 'credit_limit',
                'credit_limit_used(%)', 'credit_score', 'yearly_debt_payments',
                'prev_defaults', 'default_in_last_6months', 'gender',
                'owns_car', 'migrant_worker'
            ]
            X_existing = user_row[features].copy()

            # Label Encoding
            categorical_cols = ['occupation_type', 'gender', 'owns_car', 'owns_house', 'migrant_worker']
            for col in categorical_cols:
                X_existing[col] = X_existing[col].astype(str)
                le = joblib.load(f"le_{col}.pkl")
                X_existing[col] = le.transform(X_existing[col])

            # Feature Engineering
            X_existing['income_per_member'] = X_existing['net_yearly_income'] / (X_existing['total_family_members'] + 1)

            # Impute
            imputer = SimpleImputer(strategy='median')
            X_existing = pd.DataFrame(imputer.fit_transform(X_existing), columns=X_existing.columns)

            prob = existing_user_model.predict_proba(X_existing)[0][1]
            st.metric("Predicted Default Probability", f"{prob:.2%}")
            if prob < 0.2:
                st.success("✅ Credit Approved")
            else:
                st.error("❌ Credit Rejected")
        else:
            st.warning("❗ Customer ID not found. Please check and try again.")
