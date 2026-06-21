
# AI Resume Screening System

An AI-powered Resume Screening System built using **Python, Machine Learning, NLP, and Streamlit**.

This project automatically analyzes resumes, extracts skills, and predicts the suitable job category using a trained machine learning model.

---

## Features

- Upload Resume (PDF/TXT)
- Resume text extraction
- Skill extraction
- Resume category prediction
- TF-IDF based text processing
- Machine Learning classification
- Interactive Streamlit web application
- Fast prediction using saved ML models

---

## Demo

The application allows users to upload a resume and get:

- Predicted job category
- Extracted skills
- Resume analysis result

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLP (Natural Language Processing)
- TF-IDF Vectorization
- Pandas
- NumPy
- Joblib

---

## Project Structure

```
AI_Resume_Screening_System
│
├── app.py
│
├── train.py
│
├── model_training.py
│
├── preprocessing.py
│
├── dataset/
│
├── resume_classifier.pkl
│
├── tfidf_vectorizer.pkl
│
├── label_encoder.pkl
│
├── requirements.txt
│
└── README.md
```

---

## Machine Learning Workflow

1. Collect resume dataset

2. Data preprocessing

3. Clean resume text

4. Convert text into numerical features using TF-IDF

5. Train machine learning model

6. Save trained model using Joblib

7. Load model in Streamlit application

8. Predict resume category

---

## Installation

### Clone Repository

```bash
git clone https://github.com/husnamea-it/AI_Resume_Screening_System.git
```

### Open Project Folder

```bash
cd AI_Resume_Screening_System
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Run Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Model Files

The project uses saved machine learning files:

- `resume_classifier.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`

These files are loaded to perform resume prediction.

---

## Screenshots

Add your application screenshots here.

Example:

```
![App Screenshot](screenshot.png)
```

---

## Future Improvements

- Add AI-based resume ranking
- Add job description matching
- Improve skill extraction
- Add more resume categories
- Deploy as a web application

---

## Author

**Husna**

---

## License

This project is created for educational and portfolio purposes.
