import streamlit as st
import joblib
import PyPDF2
import re


# -----------------------------
# Load Models
# -----------------------------

model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")


# -----------------------------
# Page Setup
# -----------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄"
)

st.title("📄 AI Resume Screening System")

st.write(
    "Upload resume(s), add job description and check match score."
)


# -----------------------------
# Functions
# -----------------------------

def extract_text(file):

    text = ""

    if file.name.endswith(".pdf"):

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            text += page.extract_text()

    elif file.name.endswith(".txt"):

        text = file.read().decode("utf-8")

    return text



def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text



def extract_skills(text):

    skills = [
        "python",
        "machine learning",
        "deep learning",
        "data science",
        "sql",
        "pandas",
        "numpy",
        "tensorflow",
        "scikit-learn",
        "streamlit",
        "nlp",
        "flask",
        "git",
        "github",
        "html",
        "css",
        "javascript"
    ]

    found = []

    text = text.lower()

    for skill in skills:

        if skill in text:
            found.append(skill)

    return found



def match_score(resume, job):

    resume_words = set(
        clean_text(resume).split()
    )

    job_words = set(
        clean_text(job).split()
    )


    if len(job_words) == 0:
        return 0


    matched = resume_words.intersection(
        job_words
    )


    score = (
        len(matched)
        /
        len(job_words)
    ) * 100


    return round(score,2)



# -----------------------------
# Job Description
# -----------------------------

job_description = st.text_area(
    "📌 Paste Job Description",
    height=200
)



# -----------------------------
# Upload Resume
# -----------------------------

uploaded_files = st.file_uploader(
    "📄 Upload Resume(s)",
    type=["pdf","txt"],
    accept_multiple_files=True
)



# -----------------------------
# Processing
# -----------------------------

if uploaded_files:


    for file in uploaded_files:


        st.divider()

        st.subheader(
            file.name
        )


        resume_text = extract_text(file)


        cleaned = clean_text(
            resume_text
        )


        if cleaned == "":

            st.error(
                "Could not read resume"
            )

            continue



        # Prediction

        features = vectorizer.transform(
            [cleaned]
        )


        prediction = model.predict(
            features
        )


        category = encoder.inverse_transform(
            prediction
        )[0]



        st.write(
            "### 🎯 Predicted Category"
        )

        st.success(
            category
        )



        # Skills

        skills = extract_skills(
            cleaned
        )


        st.write(
            "### 🛠 Skills Found"
        )


        if skills:

            st.info(
                ", ".join(skills)
            )

        else:

            st.warning(
                "No skills found"
            )



        # Match Score

        if job_description:


            score = match_score(
                cleaned,
                job_description
            )


            st.write(
                "### 📊 Resume Match Score"
            )


            st.success(
                f"{score}%"
            )


        else:

            st.warning(
                "Add job description to calculate match score"
            )