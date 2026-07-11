import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Mental Health Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Student Mental Health Risk Predictor")


model = joblib.load("mental_health_model.pkl")
age = st.number_input("Age", 17, 40, 20)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Marital Status",
    ["No", "Yes"]
)

year = st.selectbox(
    "Year of Study",
    ["Year 1", "Year 2", "Year 3", "Year 4"]
)

cgpa = st.selectbox(
    "CGPA",
    [
        "0 - 1.99",
        "2.00 - 2.49",
        "2.50 - 2.99",
        "3.00 - 3.49",
        "3.50 - 4.00"
    ]
)
course = st.selectbox(
    "Academic Field",
    [
       "Computer Science & IT",
        "Engineering",
        "Business",
        "Health Sciences",
        "Social Sciences",
        "Natural Sciences",
        "Law",
        "Education",
        "Other"
    ]
)


predict = st.button("Predict Depression Risk")
if predict:
   

    gender = 1 if gender == "Female" else 0
    married = 1 if married == "Yes" else 0

    year_map = {
        "Year 1": 1,
        "Year 2": 2,
        "Year 3": 3,
        "Year 4": 4
    }

    cgpa_map = {
        "0 - 1.99": 1,
        "2.00 - 2.49": 2,
        "2.50 - 2.99": 3,
        "3.00 - 3.49": 4,
        "3.50 - 4.00": 5
    }

    data = {
        "Age": age,
        "Gender": gender,
        "Married": married,
        "Year": year_map[year],
        "CGPA": cgpa_map[cgpa]
    }

    course_columns = [
         "Course_Business",
    "Course_Computer Science & IT",
    "Course_Education",
    "Course_Engineering",
    "Course_Health Sciences",
    "Course_Law",
    "Course_Natural Sciences",
    "Course_Other",
    "Course_Social Sciences"
]
    

    # Set every course to 0
    for c in course_columns:
        data[c] = 0

    # Set the selected course to 1
    selected_course = f"Course_{course}"

    if selected_course in data:
        data[selected_course] = 1

    try:
        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.divider()

        st.header("📊 Prediction")

        if prediction == 1:
            st.error("🔴 Prediction: Depression Risk")
            confidence = probability[1]
        else:
            st.success("🟢 Prediction: Low Depression Risk")
            confidence = probability[0]

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2%}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")