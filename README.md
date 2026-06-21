# AI Resume Screening System

An AI-powered Resume Screening System built using **Python, Machine Learning, NLP, and Streamlit**.

This application analyzes resumes, extracts skills, and predicts the suitable job category using a trained Machine Learning model.

---

# 🚀 Live Demo

Streamlit Application:

https://airesumescreeningsystem-yedss29py8sdsttobgu52x.streamlit.app

---

# Features

✅ Upload Resume (PDF/TXT)  
✅ Resume text extraction  
✅ Skill extraction  
✅ Resume category prediction  
✅ TF-IDF based text processing  
✅ Machine Learning classification  
✅ Interactive Streamlit interface  
✅ Fast prediction using saved ML models  

---

# Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLP
- TF-IDF Vectorization
- Pandas
- NumPy
- Joblib

---

# Project Structure

```
AI_Resume_Screening_System
│
├── app.py
├── train.py
├── model_training.py
├── preprocessing.py
│
├── dataset/
│
├── resume_classifier.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
│
├── requirements.txt
└── README.md
```

---

# How It Works

## 1. Data Collection

Resume dataset is collected and prepared.

## 2. Data Preprocessing

Resume text is cleaned and prepared for training.

## 3. Text Vectorization

TF-IDF converts resume text into numerical features.

## 4. Model Training

Machine Learning model learns from resume data.

## 5. Prediction

User uploads resume → model predicts job category.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/husnamea-it/AI_Resume_Screening_System.git
```

Move into project folder:

```bash
cd AI_Resume_Screening_System
```

Install libraries:

```bash
pip install -r requirements.txt
```

---

# Run Locally

Start the application:

```bash
streamlit run app.py
```

The application opens in your browser.

---

# Model Files

This project uses trained files:

- resume_classifier.pkl
- tfidf_vectorizer.pkl
- label_encoder.pkl

These files are loaded for prediction.

---

# Deployment

The project is deployed using:

- GitHub
- Streamlit Cloud

---

# Future Improvements

- Resume ranking system
- Job description matching
- AI-based recommendations
- More resume categories
- Better skill extraction

---

# Author

Husna

---

# License

This project is created for educational and portfolio purposes.
