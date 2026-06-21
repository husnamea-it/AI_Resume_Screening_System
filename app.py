import streamlit as st
import joblib
import PyPDF2
import re


# -------------------------------
# Load saved ML files
# -------------------------------

model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")


# -------------------------------
# Page Setup
# -------------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄"
)

st.title("📄 AI Resume Screening System")

st.write(
    "Upload multiple resumes and get predicted job categories."
)


# -------------------------------
# Resume Text Extraction
# -------------------------------

def extract_text(file):

    text = ""

    if file.name.endswith(".pdf"):

        pdf_reader = PyPDF2.PdfReader(file)

        for page in pdf_reader.pages:
            text += page.extract_text()

    elif file.name.endswith(".txt"):

        text = file.read().decode("utf-8")


    return text



# -------------------------------
# Text Cleaning
# -------------------------------

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



# -------------------------------
# Skills Extraction
# -------------------------------

skills_list = [
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


def extract_skills(text):

    found = []

    text = text.lower()

    for skill in skills_list:

        if skill in text:
            found.append(skill)

    return found



# -------------------------------
# Upload Multiple Resumes
# -------------------------------

uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf", "txt"],
    accept_multiple_files=True
)



# -------------------------------
# Prediction
# -------------------------------

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} resume(s) uploaded"
    )


    for file in uploaded_files:

        st.divider()

        st.subheader(
            file.name
        )


        resume_text = extract_text(file)


        cleaned = clean_text(resume_text)


        if cleaned.strip() == "":
            st.warning(
                "Could not extract text"
            )
            continue



        # Vectorize

        features = vectorizer.transform(
            [cleaned]
        )


        # Predict

        prediction = model.predict(
            features
        )


        category = encoder.inverse_transform(
            prediction
        )[0]



        # Skills

        skills = extract_skills(
            cleaned
        )


        st.write(
            "### Predicted Category:"
        )

        st.success(
            category
        )


        st.write(
            "### Skills Found:"
        )


        if skills:

            st.write(
                ", ".join(skills)
            )

        else:

            st.write(
                "No skills detected"
            )
            