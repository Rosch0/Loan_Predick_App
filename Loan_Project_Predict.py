import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Title app
st.title("Creditworthiness Prediction")
st.write(""" Enter personal information and loan details to predict whether the loan will be granted. """)

# data
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

data = load_data("train.csv")

features = ['person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
target = 'loan_status'

X = data[features]
y = data[target]

# test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

pipeline.fit(X_train, y_train)

# Formula user
st.sidebar.header("Applicant details")
income = st.sidebar.number_input("Income of a person (USD)", min_value=0, value=50000, step=1000)
emp_length = st.sidebar.number_input("Number of years of employment", min_value=0, max_value=50, value=5, step=1)
loan_amnt = st.sidebar.number_input("Loan amount (USD)", min_value=0, value=10000, step=500)
int_rate = st.sidebar.number_input("Interest rate on the loan (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
percent_income = st.sidebar.number_input("Percentage of income allocated to the loan", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# enter data prepere
input_data = pd.DataFrame({
    'person_income': [income],
    'person_emp_length': [emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [int_rate],
    'loan_percent_income': [percent_income]
})

# Predict
prediction = pipeline.predict(input_data)[0]
prediction_proba = pipeline.predict_proba(input_data)[0][1]

# show result
st.subheader("Prediction result")
if prediction == 1:
    st.success(f"The loan will be granted. Probability: {prediction_proba:.2%}")
else:
    st.error(f"The loan will be rejected. Probability of granting: {prediction_proba:.2%}")

if st.checkbox("Show sample data"):
    st.write(data[features + [target]].head())
