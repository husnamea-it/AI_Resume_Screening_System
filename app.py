from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import joblib
import re
from PyPDF2 import PdfReader

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120
    )
    st.title("Resume AI")
    st.write(
        "Predict job category, extract skills and calculate resume match score."
    )

# ---------------- LOAD MODEL ----------------

model = joblib.load("resume_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ---------------- CLEAN TEXT ----------------

def clean_resume(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- TITLE ----------------

st.markdown("""
# 🤖 AI Resume Screening System

### Upload your resume and get AI-powered insights
""")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Choose a Resume",
    type=["pdf", "txt"]
)

cleaned_resume = ""

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

    # Resume Preview

    st.subheader("📄 Resume Preview")
    st.write(resume_text[:1000])

    # Clean Resume

    cleaned_resume = clean_resume(resume_text)

    # Predict Category

    resume_vector = tfidf.transform([cleaned_resume])

    prediction = model.predict(resume_vector)

    category = label_encoder.inverse_transform(prediction)

    st.subheader("🎯 Predicted Category")

    st.metric(
        label="Job Category",
        value=category[0]
    )

    # Skills Extraction

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

    st.subheader("🛠 Skills Found")

    if len(found_skills) > 0:

        col1, col2 = st.columns(2)

        for i, skill in enumerate(found_skills):

            if i % 2 == 0:
                col1.success(skill.title())

            else:
                col2.success(skill.title())

    else:
        st.warning("No skills found")

# ---------------- JOB DESCRIPTION ----------------

st.subheader("📋 Job Description Matching")

job_description = st.text_area(
    "Paste Job Description Here"
)

if uploaded_file is not None and job_description:

    jd_clean = clean_resume(job_description)

    resume_vector = tfidf.transform([cleaned_resume])

    jd_vector = tfidf.transform([jd_clean])

    score = cosine_similarity(
        resume_vector,
        jd_vector
    )[0][0]

    match_percent = round(score * 100, 2)

    st.subheader("📊 Resume Match Score")

    st.progress(int(match_percent))

    st.success(
        f"Resume Match Score: {match_percent}%"
    )

    if match_percent >= 80:
        st.success("🔥 Excellent Match")

    elif match_percent >= 60:
        st.info("✅ Good Match")

    elif match_percent >= 40:
        st.warning("⚠ Average Match")

    else:
        st.error("❌ Low Match")

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    "🤖 Developed by Husna | AI Resume Screening System"
)