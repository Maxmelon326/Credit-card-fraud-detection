# 🏦 FinTech Credit Risk Assessment Demo

This project simulates a real-world credit risk assessment system for a fintech company. It features a **Streamlit web application** that evaluates whether to approve or reject credit based on a user's profile. The system supports both **new applicants** and **existing customers**, mimicking a real credit approval pipeline.

## 🌐 Live Demo

👉 Try the deployed app on **Streamlit Cloud**:  
**🔗(https://your-streamlit-app-url)](https://credit-card-fraud-detection-vvzvrwbofbso9yk4tv8trm.streamlit.app/)**

- No installation required
- Runs in-browser with interactive UI
- Demo both new user and existing user assessment
  
![demo](https://github.com/user-attachments/assets/58f1e7b1-3ac5-48ac-aa6e-25686938b897)


## 🔧 Features

- ✅ **Two-model system**:
  - **New User Model**: Accepts personal information from first-time users.
  - **Existing User Model**: Identifies returning users via `customer_id`.
- 📊 **Credit Approval Decision**:
  - Predicts the probability of default and returns either:
    - "✅ Credit Approved" or
    - "❌ Credit Rejected"
- 🧠 **Trained using LightGBM** with SMOTE for class imbalance handling.
- 📁 **Deployed as an interactive app using Streamlit**.

## 📁 Dataset

- Sourced from a simulated credit card customer dataset (`train.csv` and `test.csv`).
- Contains demographic, financial, and behavioral features:
  - Age, Income, Employment Duration, Occupation, Credit Score, Previous Defaults, etc.

## 🚀 How It Works

### For New Users
- Users input age, income, occupation, etc.
- Data is preprocessed (e.g., log transform, encoding, feature engineering).
- Model predicts default probability and returns credit decision.

### For Existing Users
- Users input their `customer_id`.
- System fetches matching record from internal data.
- Preprocesses and evaluates with the existing user model.

## 📊 Model Highlights

- **Models Used**: LightGBM (binary classification)
- **Feature Engineering**:
  - Log transformation on skewed data
  - Income per family member
  - One-hot or label encoding of categorical variables
- **Imbalance Handling**: SMOTE oversampling
- **Performance Metrics**:
  - Evaluated using `classification_report` and AUC on validation set

## 🛠 Tech Stack

| Component     | Description                    |
|---------------|--------------------------------|
| Python        | Main programming language      |
| LightGBM      | ML model                       |
| Pandas/Numpy  | Data processing                |
| Streamlit     | Web app frontend               |
| Joblib        | Model and encoder persistence  |
| Imbalanced-learn | SMOTE for data balancing     |

## 📂 Repository Structure
```bash
├── streamlit_app.py # Main app script
├── new_user_model.pkl # Trained model for new users
├── existing_user_model.pkl # Trained model for existing users
├── le_occupation_type.pkl # Label encoders for categorical features
├── expected_new_user_columns.pkl # Expected input columns for new users
├── expected_existing_user_columns.pkl # Expected input columns for existing users
├── train.csv # Training dataset
├── test.csv # Test dataset (simulation)
├── requirements.txt # Package dependencies

