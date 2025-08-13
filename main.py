import streamlit as st
import pickle
import numpy as np

st.title("Heart Disease Predictor")  # Run using: streamlit run main.py

age = st.number_input("Age of the patient:", min_value=1, max_value=120)
sex = st.selectbox("Sex where 0==Female and 1==Male", [0, 1])
cp = st.number_input("Chest pain type (0–3):", min_value=0, max_value=3)
trestbps = st.number_input("Resting blood pressure (mm Hg):", min_value=80, max_value=250)
chol = st.number_input("Serum cholesterol (mg/dl):", min_value=100, max_value=600)
restecg = st.number_input("Resting electrocardiographic results (0–2):", min_value=0, max_value=2)
thalach = st.number_input("Maximum heart rate achieved:", min_value=60, max_value=220)
exang = st.selectbox("Exercise-induced angina where 1==yes and 0==No", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise relative to rest:", min_value=0.0, max_value=10.0)
slope = st.number_input("Slope of the peak exercise ST segment (0–2):", min_value=0, max_value=2, step=1)
ca = st.number_input("Number of major vessels colored by fluoroscopy (0–4):", min_value=0, max_value=4)
thal = st.number_input("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect):", min_value=1, max_value=3)

# Predict Button
if st.button("Predict"):
    # Load model and scaler
    model = pickle.load(open("heart_disease_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    # Prepare input in the same order as training
    input_data = np.array([[age, sex, cp, trestbps, chol, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.success("The model predicts that the patient **has :: heart disease**.")
    else:
        st.success("The model predicts that the patient **does :: not have heart disease**.")
