import streamlit as st
import joblib
import PyPDF2
import re
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")


# ---------- CSS ----------

st.markdown("""
<style>

.stApp{
background:#f4f7fb;
}


.header{
background:linear-gradient(90deg,#0f172a,#2563eb);
padding:35px;
border-radius:18px;
color:white;
text-align:center;
margin-bottom:30px;
}


.header h1{
font-size:42px;
margin:0;
}


.card{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0 8px 25px rgba(0,0,0,0.08);
height:100%;
}


.card h2{
color:#1e3a8a;
}


.result{
background:#ecfdf5;
border-left:6px solid #10b981;
padding:25px;
border-radius:15px;
font-size:22px;
}


.metric{
background:white;
padding:20px;
border-radius:15px;
text-align:center;
box-shadow:0 5px 15px #ddd;
}


</style>
""",unsafe_allow_html=True)



# Header

st.markdown("""
<div class="header">

<h1>📄 AI Resume Screening System</h1>

<p>
Smart Resume Analysis • Skill Extraction • Job Matching
</p>

</div>
""",unsafe_allow_html=True)



# Sidebar

with st.sidebar:

    st.title("⚙️ About")

    st.info(
    """
    This AI system analyzes resumes
    and compares them with job
    descriptions using Machine Learning.
    """
    )


    st.write("### Features")

    st.write("✔ Resume Classification")
    st.write("✔ Skill Detection")
    st.write("✔ Job Match Score")
    st.write("✔ PDF Support")




# Main

col1,col2 = st.columns(2)


with col1:

    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )

    st.markdown("## 📤 Upload Resume")

    file = st.file_uploader(
        "Choose PDF or TXT",
        type=["pdf","txt"]
    )

    st.markdown("</div>",
    unsafe_allow_html=True)




with col2:

    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )

    st.markdown("## 💼 Job Description")

    job = st.text_area(
        "Paste job requirements",
        height=200,
        placeholder="Example: Python developer with ML experience..."
    )

    st.markdown("</div>",
    unsafe_allow_html=True)




def read_file(file):

    if file.type=="application/pdf":

        pdf=PyPDF2.PdfReader(file)

        text=""

        for p in pdf.pages:
            text+=p.extract_text()

        return text

    return file.read().decode()



if file and job:


    resume=read_file(file)


    cleaned=re.sub(
        "[^a-zA-Z ]",
        "",
        resume
    )


    vector=vectorizer.transform(
        [cleaned]
    )


    prediction=model.predict(vector)

    category=encoder.inverse_transform(
        prediction
    )[0]


    score=cosine_similarity(
        vectorizer.transform([resume]),
        vectorizer.transform([job])
    )[0][0]*100



    st.markdown("---")


    st.markdown("## 📊 Analysis Result")


    a,b,c=st.columns(3)


    with a:
        st.markdown(
        f"""
        <div class="metric">

        🎯 Category<br>
        <b>{category}</b>

        </div>
        """,
        unsafe_allow_html=True
        )


    with b:
        st.markdown(
        f"""
        <div class="metric">

        📈 Match<br>
        <b>{score:.2f}%</b>

        </div>
        """,
        unsafe_allow_html=True
        )


    with c:

        st.markdown(
        """
        <div class="metric">

        🤖 AI<br>
        <b>Active</b>

        </div>
        """,
        unsafe_allow_html=True
        )



    st.success("Resume analysis completed successfully 🚀")