# 🫀 Heart Disease Prediction

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A lightweight clinical-screening web app that predicts heart disease risk from key patient parameters.**

[🚀 Live Demo](https://nitesh864-heart-disease-prediction-main-vaf80i.streamlit.app/) • [💻 GitHub Repository](https://github.com/NITESH864/Heart_disease_prediction) • [📚 Documentation](https://github.com/NITESH864/Heart_disease_prediction#readme)

</div>

---

## ✨ Project Overview
This project combines a trained machine learning model and an intuitive Streamlit interface to provide fast, explainable, point-of-care style heart disease risk predictions.

### 💡 Value Proposition
- Supports quick preliminary risk checks
- Improves consistency with standardized model inputs
- Offers a simple UI suitable for demos, students, and prototype healthcare workflows

---

## 🧰 Technology Stack
- **Frontend / UI:** Streamlit
- **Machine Learning:** scikit-learn (Logistic Regression)
- **Data Processing:** NumPy, pandas
- **Visualization / Analysis:** matplotlib, seaborn

---

## 📊 Dataset Documentation
The model uses a structured heart dataset (`heart.csv`) with the following features:

| Feature | Description | Typical Values |
|---|---|---|
| `age` | Age of patient | 1–120 |
| `sex` | Biological sex | 0 = Female, 1 = Male |
| `cp` | Chest pain type | 0–3 |
| `trestbps` | Resting blood pressure (mm Hg) | 80–250 |
| `chol` | Serum cholesterol (mg/dl) | 100–600 |
| `restecg` | Resting ECG results | 0–2 |
| `thalach` | Maximum heart rate achieved | 60–220 |
| `exang` | Exercise-induced angina | 0 = No, 1 = Yes |
| `oldpeak` | ST depression induced by exercise | 0.0–10.0 |
| `slope` | Slope of peak ST segment | 0–2 |
| `ca` | Number of major vessels by fluoroscopy | 0–4 |
| `thal` | Thalassemia status | 1–3 |
| `target` | Label | 0 = No disease, 1 = Disease |

---

## 🧠 Model & Performance
> **Model:** Logistic Regression  
> **Preprocessing:** StandardScaler  
> **Train/Test Split:** 80/20  
> **Observed Accuracy:** ~85% (as reported in project notebook)

Artifacts:
- `heart_disease_model.pkl`
- `scaler.pkl`

---

## 🌟 Features
### 🎨 User Experience
- Modern Streamlit layout with grouped sections
- Helpful parameter tooltips and validation feedback
- Color-coded risk output for quick interpretation

### 🤖 Prediction Workflow
- Real-time risk inference from form inputs
- Preprocessing aligned with trained model pipeline
- Fast local execution and easy cloud deployment

---

## ⚙️ Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/NITESH864/Heart_disease_prediction.git
   cd Heart_disease_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run main.py
   ```

---

## ▶️ Usage
1. Enter patient demographics and health parameters.
2. Click **Predict Heart Disease Risk**.
3. Review the result card and follow medical guidance.

### 🖼️ Screenshots / Demo
- Live app: https://nitesh864-heart-disease-prediction-main-vaf80i.streamlit.app/
- Add screenshots here if you want visual documentation in the repo.

---

## 🧪 Troubleshooting
- **`ModuleNotFoundError`**: Run `pip install -r requirements.txt` again.
- **Model file errors**: Ensure `heart_disease_model.pkl` and `scaler.pkl` are in the project root.
- **Streamlit not launching**: Try `python -m streamlit run main.py`.

---

## 🤝 Contributing
Contributions are welcome.
1. Fork the repository
2. Create a feature branch
3. Make focused changes with clear commit messages
4. Open a pull request with a concise description

---

## 📄 License & Acknowledgments
- Licensed under the **MIT License**.
- Thanks to the open-source data science and healthcare ML communities for tools and inspiration.
