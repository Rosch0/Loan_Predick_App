import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# CSS
st.markdown(
    """
    <style>
    .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="centered-container">', unsafe_allow_html=True)

# App title
st.title("Creditworthiness Prediction")
st.write("Enter personal information and loan details to predict whether the loan will be granted.")

# Function to load data
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

data = load_data("train.csv")

features = ['person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
target = 'loan_status'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

pipeline.fit(X_train, y_train)

# Input form for user details
st.subheader("Applicant Details")
income = st.number_input("Income of a person (USD)", min_value=0, value=50000, step=1000)
emp_length = st.number_input("Number of years of employment", min_value=0, max_value=50, value=5, step=1)
loan_amnt = st.number_input("Loan amount (USD)", min_value=0, value=10000, step=500)
int_rate = st.number_input("Interest rate on the loan (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
percent_income = st.number_input("Percentage of income allocated to the loan", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# Input data for the model
input_data = pd.DataFrame({
    'person_income': [income],
    'person_emp_length': [emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [int_rate],
    'loan_percent_income': [percent_income]
})

# Predict button
if st.button("Predict"):
    # Prediction
    prediction = pipeline.predict(input_data)[0]
    prediction_proba = pipeline.predict_proba(input_data)[0][1]

    # Display result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"The loan will be granted. Probability: {prediction_proba:.2%}")
    else:
        st.error(f"The loan will be rejected. Probability of granting: {prediction_proba:.2%}")

# Option to display sample data
if st.checkbox("Show sample data"):
    st.write(data[features + [target]].head())

# Closing the container
st.markdown('</div>', unsafe_allow_html=True)
