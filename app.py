from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import joblib
import re
from PyPDF2 import PdfReader

# Load saved model
model = joblib.load("resume_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Clean resume text
def clean_resume(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Website title
st.title("🤖 AI Resume Screening System")

st.write("Upload a resume and predict its job category.")

# Upload text file
uploaded_file = st.file_uploader(
    "Choose a Resume",
    type=["pdf", "txt"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        pdf = PdfReader(uploaded_file)
        resume_text = ""

        for page in pdf.pages:
            text = page.extract_text()
            if text:
                resume_text += text

    else:
        resume_text = uploaded_file.read().decode("utf-8")

    st.subheader("Resume Preview")
    st.write(resume_text[:500])  # Show first 500 characters

    cleaned_resume = clean_resume(resume_text)
    resume_vector = tfidf.transform([cleaned_resume])

    prediction = model.predict(resume_vector)
    category = label_encoder.inverse_transform(prediction)

    st.success(f"Predicted Category: {category[0]}")
    skills_list = [
        "python",
        "sql",
        "machine learning",
        "deep learning",
        "pandas",
        "numpy",
        "java",
        "html",
        "css",
        "javascript",
        "react",
        "django",
        "flask",
        "git"
    ]

    found_skills = []

    for skill in skills_list:
        if skill in cleaned_resume:
            found_skills.append(skill)

    st.subheader("Skills Found")

    for skill in found_skills:
        st.write("✔", skill.title())
    st.subheader("Job Description Matching")

job_description = st.text_area(
    "Paste Job Description Here"
)

if job_description:

    jd_clean = clean_resume(job_description)

    resume_vector = tfidf.transform([cleaned_resume])

    jd_vector = tfidf.transform([jd_clean])

    score = cosine_similarity(
        resume_vector,
        jd_vector
    )[0][0]

    match_percent = round(score * 100, 2)

    st.success(
        f"Resume Match Score: {match_percent}%"
    )