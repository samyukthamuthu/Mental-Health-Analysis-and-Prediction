import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Mental Health Prediction")

# ---------- TEXT INPUTS ----------

gender_text = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input("Age", 1, 100, 18)

city_text = st.selectbox(
    "City",
    ["Chennai", "Coimbatore", "Madurai", "Salem", "Other"]
)

profession_text = st.selectbox(
    "Profession",
    ["Student", "Working Professional"]
)

academic_pressure = st.slider("Academic Pressure", 0, 10, 1)

work_pressure = st.slider("Work Pressure", 0, 10, 1)

cgpa = st.number_input("CGPA", 0.0, 10.0, 5.0)

study_satisfaction = st.slider("Study Satisfaction", 0, 10, 5)

job_satisfaction = st.slider("Job Satisfaction", 0, 10, 5)

sleep_duration = st.slider("Sleep Duration", 0, 12, 7)

diet_text = st.selectbox(
    "Dietary Habits",
    ["Healthy", "Moderate", "Unhealthy"]
)

degree_text = st.selectbox(
    "Degree",
    ["UG", "PG", "Diploma", "Other"]
)

suicidal_text = st.selectbox(
    "Have you ever had suicidal thoughts?",
    ["Yes", "No"]
)

work_hours = st.slider("Work/Study Hours", 0, 24, 6)

financial_stress = st.slider("Financial Stress", 0, 10, 5)

family_text = st.selectbox(
    "Family History of Mental Illness",
    ["Yes", "No"]
)


# ---------- CONVERT TO NUMBERS ----------

gender = 1 if gender_text == "Male" else 0
city = 1
profession = 1 if profession_text == "Student" else 0

diet = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}[diet_text]

degree = 1

suicidal = 1 if suicidal_text == "Yes" else 0

family = 1 if family_text == "Yes" else 0


# ---------- PREDICT ----------

if st.button("Predict"):

    data = pd.DataFrame([[
        gender,
        age,
        city,
        profession,
        academic_pressure,
        work_pressure,
        cgpa,
        study_satisfaction,
        job_satisfaction,
        sleep_duration,
        diet,
        degree,
        suicidal,
        work_hours,
        financial_stress,
        family
    ]])

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("Result: Depressed")
        st.write("Opinion: User may need mental health support")
    else:
        st.success("Result: Not Depressed")
        st.write("Opinion: User seems mentally stable")
