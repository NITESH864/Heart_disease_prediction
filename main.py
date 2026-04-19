import pickle

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .app-card {
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        background: rgba(127, 127, 127, 0.06);
    }
    .result-box {
        border-radius: 14px;
        padding: 1rem;
        margin-top: 0.8rem;
        font-weight: 600;
    }
    .risk {
        border: 1px solid rgba(255, 0, 0, 0.35);
        background: rgba(255, 0, 0, 0.12);
        color: #ff4b4b;
    }
    .safe {
        border: 1px solid rgba(0, 180, 0, 0.35);
        background: rgba(0, 180, 0, 0.12);
        color: #00a86b;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }
    .footer-note {
        margin-top: 1.5rem;
        font-size: 0.92rem;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = pickle.load(open("heart_disease_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler


st.title("🫀 Heart Disease Prediction App")
st.caption("A clinical support prototype — not a substitute for professional diagnosis.")

with st.container(border=True):
    st.subheader("👤 Patient Information")
    patient_name = st.text_input("Patient name", placeholder="e.g., Alex Kumar")
    visit_id = st.text_input("Visit ID", placeholder="Optional")

st.progress(33, text="Step 1 of 3: Enter patient and health details")
st.divider()

with st.expander("🧾 Demographics", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45,
            help="Patient age in years.",
        )
        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="Biological sex used by the model.",
        )
    with col2:
        cp = st.number_input(
            "Chest pain type (0-3)",
            min_value=0,
            max_value=3,
            value=1,
            help="0: Typical angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic.",
        )
        exang = st.selectbox(
            "Exercise-induced angina",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether chest pain occurs during exercise.",
        )

with st.expander("🩺 Cardiovascular Measurements", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=80,
            max_value=250,
            value=120,
            help="Blood pressure measured at rest.",
        )
        chol = st.number_input(
            "Serum cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=220,
            help="Total cholesterol in mg/dl.",
        )
    with col2:
        thalach = st.number_input(
            "Maximum heart rate achieved",
            min_value=60,
            max_value=220,
            value=150,
            help="Peak heart rate reached during testing.",
        )
        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            help="ST depression induced by exercise relative to rest.",
        )

with st.expander("🧪 ECG & Test Results", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        restecg = st.number_input(
            "Resting ECG results (0-2)",
            min_value=0,
            max_value=2,
            value=1,
            help="Resting electrocardiographic classification.",
        )
        slope = st.number_input(
            "Slope of peak exercise ST segment (0-2)",
            min_value=0,
            max_value=2,
            value=1,
            step=1,
            help="Slope category from stress test.",
        )
    with col2:
        ca = st.number_input(
            "Major vessels colored by fluoroscopy (0-4)",
            min_value=0,
            max_value=4,
            value=0,
            help="Count of major vessels visualized by fluoroscopy.",
        )
        thal = st.number_input(
            "Thalassemia (1-3)",
            min_value=1,
            max_value=3,
            value=2,
            help="1: Normal, 2: Fixed defect, 3: Reversible defect.",
        )

warnings = []
if trestbps > 180:
    warnings.append("High resting blood pressure entered.")
if chol > 300:
    warnings.append("High serum cholesterol entered.")
if oldpeak > 4:
    warnings.append("Severe ST depression entered.")
if age < 18:
    warnings.append("Model interpretation may be less reliable for pediatric ages.")

if warnings:
    st.warning("⚠️ Input validation feedback:\n- " + "\n- ".join(warnings))
else:
    st.success("✅ Inputs look within the model's expected ranges.")

st.progress(66, text="Step 2 of 3: Ready for model inference")

predict_clicked = st.button("🔍 Predict Heart Disease Risk", type="primary")

if predict_clicked:
    model, scaler = load_artifacts()
    input_data = np.array(
        [[age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        dtype=float,
    )
    input_scaled = scaler.transform(input_data)
    prediction = int(model.predict(input_scaled)[0])

    st.progress(100, text="Step 3 of 3: Prediction complete")
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown(
        f"**Patient:** {patient_name or 'Not provided'} &nbsp;&nbsp;|&nbsp;&nbsp; **Visit ID:** {visit_id or 'Not provided'}"
    )

    if prediction == 1:
        st.markdown(
            '<div class="result-box risk">🔴 Elevated risk detected. Please consult a cardiologist for detailed evaluation.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-box safe">🟢 Lower risk indicated by the model. Continue healthy habits and regular checkups.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div class="footer-note">ℹ️ <strong>Disclaimer:</strong> This tool supports educational and screening workflows only and must not replace professional medical judgment.</div>',
    unsafe_allow_html=True,
)
