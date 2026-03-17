import os
import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Mental Health Depression Prediction",
    page_icon="🧠",
    layout="centered",
)

BUNDLE_PATH = "depression_model_bundle.pkl"

FIELD_LABELS = {
    "Gender": "Gender",
    "Age": "Age",
    "City": "City",
    "Profession": "Profession",
    "Academic Pressure": "Academic Pressure",
    "Work Pressure": "Work Pressure",
    "CGPA": "CGPA",
    "Study Satisfaction": "Study Satisfaction",
    "Job Satisfaction": "Job Satisfaction",
    "Sleep Duration": "Sleep Duration (hours)",
    "Dietary Habits": "Dietary Habits",
    "Degree": "Degree",
    "Have you ever had suicidal thoughts ?": "Have you ever had suicidal thoughts?",
    "Work/Study Hours": "Work / Study Hours",
    "Financial Stress": "Financial Stress",
    "Family History of Mental Illness": "Family History of Mental Illness",
}

FIELD_HELP = {
    "Academic Pressure": "Rate from 0 to 10, where 0 means no pressure and 10 means extremely high pressure.",
    "Work Pressure": "Rate from 0 to 10 based on your work-related stress.",
    "Study Satisfaction": "Rate from 0 to 10 based on how satisfied you feel with your studies.",
    "Job Satisfaction": "Rate from 0 to 10 based on how satisfied you feel with your work.",
    "Sleep Duration": "Average number of hours you sleep.",
    "CGPA": "Enter your academic score on a 10-point scale if applicable.",
    "Have you ever had suicidal thoughts ?": "Select the option that best matches your response.",
    "Work/Study Hours": "Average number of hours spent on work or study in a day.",
    "Financial Stress": "Rate from 0 to 10 based on your financial stress level.",
}

NUMERIC_SLIDER_FIELDS = {
    "Academic Pressure",
    "Work Pressure",
    "Study Satisfaction",
    "Job Satisfaction",
    "Financial Stress",
}


def prettify_option(value: str) -> str:
    value = str(value).strip()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)

    lower_map = {
        "yes": "Yes",
        "no": "No",
        "male": "Male",
        "female": "Female",
        "student": "Student",
        "working professional": "Working Professional",
        "healthy": "Healthy",
        "moderate": "Moderate",
        "unhealthy": "Unhealthy",
    }
    if value.lower() in lower_map:
        return lower_map[value.lower()]

    return value.title()


@st.cache_resource

def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        return None
    return joblib.load(BUNDLE_PATH)



def build_input_form(bundle):
    st.subheader("Enter your details")
    st.caption("Use clear wording for category fields and numbers only where needed.")

    label_encoders = bundle["label_encoders"]
    feature_columns = bundle["feature_columns"]
    user_values = {}

    col_left, col_right = st.columns(2)

    for i, col in enumerate(feature_columns):
        container = col_left if i % 2 == 0 else col_right
        label = FIELD_LABELS.get(col, col)
        help_text = FIELD_HELP.get(col)

        with container:
            if col in label_encoders:
                raw_options = list(label_encoders[col].classes_)
                display_options = [prettify_option(opt) for opt in raw_options]
                display_to_raw = dict(zip(display_options, raw_options))

                default_index = 0
                if "No" in display_options:
                    default_index = display_options.index("No")
                elif "Male" in display_options:
                    default_index = display_options.index("Male")
                elif "Student" in display_options:
                    default_index = display_options.index("Student")

                chosen_display = st.selectbox(
                    label,
                    display_options,
                    index=default_index,
                    help=help_text,
                    key=col,
                )
                user_values[col] = display_to_raw[chosen_display]

            elif col == "Age":
                user_values[col] = st.number_input(
                    label,
                    min_value=10,
                    max_value=100,
                    value=21,
                    step=1,
                    help=help_text,
                )
            elif col in NUMERIC_SLIDER_FIELDS:
                user_values[col] = st.slider(
                    label,
                    min_value=0,
                    max_value=10,
                    value=5,
                    step=1,
                    help=help_text,
                )
            elif col == "CGPA":
                user_values[col] = st.number_input(
                    label,
                    min_value=0.0,
                    max_value=10.0,
                    value=7.0,
                    step=0.01,
                    help=help_text,
                )
            elif col == "Sleep Duration":
                user_values[col] = st.slider(
                    label,
                    min_value=0,
                    max_value=12,
                    value=7,
                    step=1,
                    help=help_text,
                )
            elif col == "Work/Study Hours":
                user_values[col] = st.slider(
                    label,
                    min_value=0,
                    max_value=24,
                    value=8,
                    step=1,
                    help=help_text,
                )
            else:
                user_values[col] = st.number_input(
                    label,
                    value=0.0,
                    help=help_text,
                )

    return user_values



def encode_input(user_values, bundle):
    label_encoders = bundle["label_encoders"]
    encoded = {}

    for col, value in user_values.items():
        if col in label_encoders:
            encoded[col] = label_encoders[col].transform([value])[0]
        else:
            encoded[col] = value

    return pd.DataFrame([encoded])[bundle["feature_columns"]]



def show_result(prediction, probability=None):
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.error("**Result: Depressed**")
        st.write(
            "**Opinion:** The input pattern shows signs associated with depression risk. "
            "This output is only a machine learning prediction and not a medical diagnosis."
        )
        st.info(
            "Please consider speaking with a mental health professional, counselor, trusted friend, or family member if support is needed."
        )
    else:
        st.success("**Result: Not Depressed**")
        st.write(
            "**Opinion:** The input pattern suggests a lower depression risk based on the trained model. "
            "Continue maintaining healthy routines and self-care habits."
        )

    if probability is not None:
        st.caption(f"Estimated depression probability: {probability:.2%}")



def main():
    st.title("🧠 Mental Health Depression Prediction")
    st.write(
        "This web app predicts whether a person may be at risk of depression based on lifestyle, "
        "academic, work, and mental health related inputs."
    )

    bundle = load_bundle()

    if bundle is None:
        st.error("Model bundle not found.")
        st.info(
            "Keep `depression_model_bundle.pkl` in the same folder as `app.py`. "
            "Create it first by running `train_and_save_model.py`."
        )
        st.stop()

    with st.expander("About the input values"):
        st.write(
            "Choose word-based options for category fields such as Gender, Profession, City, Degree, "
            "Dietary Habits, and family history. Use numbers only for age, pressure scores, CGPA, sleep duration, and work/study hours."
        )

    user_values = build_input_form(bundle)

    if st.button("Predict", use_container_width=True):
        input_df = encode_input(user_values, bundle)
        prediction = bundle["model"].predict(input_df)[0]

        try:
            probability = bundle["model"].predict_proba(input_df)[0][1]
        except Exception:
            probability = None

        show_result(prediction, probability)

    st.markdown("---")
    st.caption("For educational use only. This app does not replace professional mental health advice or diagnosis.")


if __name__ == "__main__":
    main()
