from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "future_leader_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.joblib"


DEFAULT_FEATURES = [
    "department",
    "job_level",
    "education",
    "performance_score",
    "engagement_score",
    "years_experience",
    "years_at_company",
    "leadership_trait_score",
    "peer_feedback_score",
    "manager_feedback_score",
    "innovation_score",
    "learning_hours",
    "projects_led",
    "promotion_last_3yrs",
    "absenteeism_days",
    "attrition_risk_score",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


st.set_page_config(page_title="Future Leader Predictor", layout="wide")
st.title("Identifying Future Leaders Through Data Analytics")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files not found. Run `python future_leader_pipeline.py` first.")
    st.stop()

model, metadata = load_artifacts()
feature_columns = metadata.get("feature_columns", DEFAULT_FEATURES)

with st.sidebar:
    st.header("Employee Inputs")
    department = st.selectbox("Department", ["Sales", "Engineering", "HR", "Finance", "Operations", "Marketing", "Product"])
    job_level = st.selectbox("Job Level", ["Junior", "Mid", "Senior", "Lead", "Manager"], index=2)
    education = st.selectbox("Education", ["Bachelor", "Master", "PhD", "Diploma"], index=1)

    performance_score = st.slider("Performance Score", 0, 100, 85)
    engagement_score = st.slider("Engagement Score", 0, 100, 82)
    years_experience = st.number_input("Years Experience", min_value=0.0, max_value=40.0, value=7.0, step=0.5)
    years_at_company = st.number_input("Years at Company", min_value=0.0, max_value=30.0, value=4.0, step=0.5)
    leadership_trait_score = st.slider("Leadership Trait Score", 0, 100, 88)
    peer_feedback_score = st.slider("Peer Feedback Score", 1.0, 5.0, 4.3, 0.1)
    manager_feedback_score = st.slider("Manager Feedback Score", 1.0, 5.0, 4.4, 0.1)
    innovation_score = st.slider("Innovation Score", 0, 100, 80)
    learning_hours = st.number_input("Learning Hours", min_value=0.0, max_value=200.0, value=45.0, step=1.0)
    projects_led = st.number_input("Projects Led", min_value=0, max_value=30, value=4, step=1)
    promotion_last_3yrs = st.selectbox("Promoted in Last 3 Years", [0, 1], index=1)
    absenteeism_days = st.number_input("Absenteeism Days", min_value=0, max_value=60, value=3, step=1)
    attrition_risk_score = st.slider("Attrition Risk Score", 0, 100, 22)


employee_data = {
    "department": department,
    "job_level": job_level,
    "education": education,
    "performance_score": performance_score,
    "engagement_score": engagement_score,
    "years_experience": years_experience,
    "years_at_company": years_at_company,
    "leadership_trait_score": leadership_trait_score,
    "peer_feedback_score": peer_feedback_score,
    "manager_feedback_score": manager_feedback_score,
    "innovation_score": innovation_score,
    "learning_hours": learning_hours,
    "projects_led": projects_led,
    "promotion_last_3yrs": promotion_last_3yrs,
    "absenteeism_days": absenteeism_days,
    "attrition_risk_score": attrition_risk_score,
}

employee_df = pd.DataFrame([employee_data], columns=feature_columns)
prediction = int(model.predict(employee_df)[0])
probability = float(model.predict_proba(employee_df)[0][1]) if hasattr(model.named_steps["model"], "predict_proba") else None

left, right = st.columns([1, 1])

with left:
    st.subheader("Prediction")
    if prediction == 1:
        st.success("Future Leader")
    else:
        st.warning("Not Future Leader Yet")

    if probability is not None:
        st.metric("Future Leader Probability", f"{probability:.1%}")
        st.progress(max(0.0, min(1.0, probability)))

with right:
    st.subheader("Model Comparison")
    st.dataframe(metadata["model_results"], use_container_width=True, hide_index=True)
    st.caption(f"Selected model: {metadata['best_model_name']}")

st.subheader("Employee Data Sent to Model")
st.dataframe(employee_df, use_container_width=True, hide_index=True)
