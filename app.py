import streamlit as st
import joblib
import PyPDF2
import re


# ----------------------------
# Load Model Files
# ----------------------------

model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")


# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# ----------------------------
# Header
# ----------------------------

st.title("📄 AI Resume Screening System")

st.markdown(
"""
### Smart Resume Analyzer using Machine Learning

Upload resumes, compare with job descriptions,
extract skills and get AI-based matching results.
"""
)


st.divider()


# ----------------------------
# Functions
# ----------------------------

def extract_text(file):

    text = ""

    if file.name.endswith(".pdf"):

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            text += page.extract_text()

    else:

        text = file.read().decode("utf-8")


    return text



def clean_text(text):

    text = text.lower()

    text = re.sub(
        "[^a-zA-Z ]",
        " ",
        text
    )

    return text



def extract_skills(text):

    skills = [
        "python",
        "machine learning",
        "data science",
        "sql",
        "pandas",
        "numpy",
        "tensorflow",
        "scikit-learn",
        "nlp",
        "streamlit",
        "flask",
        "git",
        "github",
        "html",
        "css",
        "javascript"
    ]


    found=[]

    text=text.lower()


    for skill in skills:

        if skill in text:
            found.append(skill)


    return found



def calculate_score(resume,job):

    resume_words=set(
        clean_text(resume).split()
    )

    job_words=set(
        clean_text(job).split()
    )


    if not job_words:
        return 0


    matched=len(
        resume_words.intersection(job_words)
    )


    return round(
        (matched/len(job_words))*100,
        2
    )



# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:

    st.header("📌 About")

    st.write(
        """
        AI Resume Screening System

        Features:
        - Resume Classification
        - Skill Extraction
        - Job Matching
        - Multiple Resume Upload
        """
    )


# ----------------------------
# Input Area
# ----------------------------

col1,col2 = st.columns(2)


with col1:

    job_description = st.text_area(
        "📝 Job Description",
        height=250,
        placeholder="Paste job description here..."
    )


with col2:

    uploaded_files = st.file_uploader(
        "📂 Upload Resume(s)",
        type=["pdf","txt"],
        accept_multiple_files=True
    )



# ----------------------------
# Results
# ----------------------------

if uploaded_files:


    st.success(
        f"{len(uploaded_files)} resume(s) uploaded"
    )


    for file in uploaded_files:


        st.divider()

        st.subheader(
            f"📄 {file.name}"
        )


        text = extract_text(file)

        cleaned = clean_text(text)



        if cleaned.strip()=="":

            st.error(
                "Unable to read file"
            )

            continue



        features = vectorizer.transform(
            [cleaned]
        )


        prediction = model.predict(
            features
        )


        category = encoder.inverse_transform(
            prediction
        )[0]



        c1,c2,c3 = st.columns(3)


        with c1:

            st.metric(
                "Prediction",
                category
            )


        with c2:

            skills=extract_skills(cleaned)

            st.metric(
                "Skills Found",
                len(skills)
            )


        with c3:

            if job_description:

                score=calculate_score(
                    cleaned,
                    job_description
                )

                st.metric(
                    "Match Score",
                    f"{score}%"
                )

            else:

                st.metric(
                    "Match Score",
                    "Add JD"
                )



        st.write(
            "### 🛠 Extracted Skills"
        )


        if skills:

            st.info(
                ", ".join(skills)
            )

        else:

            st.warning(
                "No skills detected"
            )