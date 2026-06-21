# AI Resume Screening System

An AI-powered Resume Screening System built using **Python, Machine Learning, NLP, and Streamlit**.

This application analyzes resumes, extracts skills, and predicts the suitable job category using a trained Machine Learning model.

---

## 🚀 Live Application

Try the deployed application here:

https://airesumescreeningsystem-yedss29py8sdsttobgu52x.streamlit.app
---

# 🚀 Live Demo


Streamlit Application:

https://airesumescreeningsystem-yedss29py8sdsttobgu52x.streamlit.app


# Features


- Upload single or multiple resumes (PDF/TXT)
- Resume text extraction
- Resume category prediction
- Skill extraction
- Batch resume analysis
- TF-IDF based text processing
- Machine Learning classification
- Interactive Streamlit web interface
- Fast prediction using saved ML models


# Technologies Used

- Upload Resume (PDF/TXT)
- Resume text extraction
- Resume category prediction
- Skill extraction
- TF-IDF based text processing
- Machine Learning classification
- Interactive Streamlit interface
- Fast prediction using saved ML models

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLP
- Natural Language Processing (NLP)
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
## How It Works

### 1. Data Collection

Resume dataset is collected for training the model.

### 2. Data Preprocessing

Resume text is cleaned and prepared.

### 3. Feature Extraction

TF-IDF converts resume text into numerical features.

### 4. Model Training

Machine Learning model learns patterns from resume data.

### 5. Prediction

User uploads a resume and the system predicts the suitable category.
>>>>>>> f3d3669 (Update README with app link)

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
Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Locally

Start the application:
## Run Locally

Start Streamlit application:

```bash
streamlit run app.py
```

The application opens in your browser.

---

# Model Files



This project uses trained ML files:
- resume_classifier.pkl
- tfidf_vectorizer.pkl
- label_encoder.pkl

These files are loaded for prediction.

---

# Deployment

The project is deployed using:

These files are loaded to perform predictions.

---

## Deployment

The application is deployed using:

- GitHub
- Streamlit Cloud

---

# Future Improvements

- Resume ranking system
- Job description matching

- AI-based recommendations
- More resume categories
- Better skill extraction
- AI-based career recommendations
- Better skill extraction
- More resume categories
---

# Author

Husna

---

# License

This project is created for educational and portfolio purposes.