import streamlit as st
import fitz
import os
import re
import faiss
import numpy as np
import requests
from datetime import datetime

from sentence_transformers import SentenceTransformer
from docx import Document

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI PM Assistant",
    layout="wide"
)

# -----------------------------------
# ORION CLEAN UI THEME
# -----------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F5F7FB;
        color: #1E293B;
        font-family: 'Segoe UI', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1 {
        color: #1E293B;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    h2, h3 {
        color: #334155;
        font-weight: 600;
    }

    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.04);
    }

    .stButton > button {
        background: linear-gradient(
            90deg,
            #34D399,
            #10B981
        );
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #10B981,
            #059669
        );
        color: white;
    }

    .stDownloadButton > button {
        background: linear-gradient(
            90deg,
            #6366F1,
            #4F46E5
        );
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        width: 100%;
    }

    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #CBD5E1;
        background-color: white;
        color: #111827;
        padding: 1rem;
    }

    section[data-testid="stFileUploader"] {
        background-color: white;
        border: 1px dashed #CBD5E1;
        border-radius: 14px;
        padding: 1rem;
    }

    section[data-testid="stFileUploader"] button {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
    }

    .streamlit-expanderHeader {
        background-color: #ECFDF5;
        color: #065F46;
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stSelectbox div[data-baseweb="select"] {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #CBD5E1;
    }

    .stSuccess {
        background-color: #DCFCE7;
        border-radius: 10px;
    }

    .stWarning {
        background-color: #FEF3C7;
        border-radius: 10px;
    }

    .stError {
        background-color: #FEE2E2;
        border-radius: 10px;
    }

    hr {
        border-top: 1px solid #E2E8F0;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# CREATE FOLDERS
# -----------------------------------

os.makedirs("repository", exist_ok=True)
os.makedirs("generated_fsds", exist_ok=True)

# -----------------------------------
# EMBEDDING MODEL
# -----------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
# SESSION STATE
# -----------------------------------

if "generated_fsd" not in st.session_state:
    st.session_state.generated_fsd = ""

if "approved_fsd" not in st.session_state:
    st.session_state.approved_fsd = ""

# -----------------------------------
# TITLE
# -----------------------------------

st.title("AI PM Assistant")

st.markdown(
    """
### AI Product Execution Workflow

Requirement → AI Draft → PM Review → QA Test Cases → Release Notes → GTM Content → Support Notes
"""
)

# -----------------------------------
# METRICS
# -----------------------------------

repository_files = os.listdir("repository")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Repository Files",
        len(repository_files)
    )

with col2:
    st.metric(
        "Generated FSDs",
        len(os.listdir("generated_fsds"))
    )

with col3:
    st.metric(
        "AI Status",
        "Active"
    )

# -----------------------------------
# TABS
# -----------------------------------

tab1, tab2, tab3 = st.tabs([
    "FSD Generation",
    "QA",
    "Release Notes"
])

# ===================================
# TAB 1 - FSD GENERATION
# ===================================

with tab1:

    # -----------------------------------
    # REPOSITORY VIEW
    # -----------------------------------

    with st.expander("View Repository Files"):

        if repository_files:

            for file in repository_files:
                st.write(file)

        else:
            st.warning("Repository is empty")

    # -----------------------------------
    # FILE UPLOAD
    # -----------------------------------

    st.subheader("Upload Additional Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    # -----------------------------------
    # REQUIREMENT
    # -----------------------------------

    st.subheader("Enter Requirement")

    requirement = st.text_area(
        "Requirement",
        height=200
    )

    analyze = st.button("Generate FSD")

    # -----------------------------------
    # GENERATE FSD
    # -----------------------------------

    if analyze:

        if not requirement:

            st.error("Please enter requirement")

        else:

            # SAVE FILES

            if uploaded_files:

                for uploaded_file in uploaded_files:

                    file_path = os.path.join(
                        "repository",
                        uploaded_file.name
                    )

                    with open(file_path, "wb") as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )

            repository_files = os.listdir(
                "repository"
            )

            all_chunks = []

            chunk_size = 800
            chunk_overlap = 100

            # -----------------------------------
            # READ FILES
            # -----------------------------------

            for file in repository_files:

                file_path = os.path.join(
                    "repository",
                    file
                )

                text = ""

                try:

                    if file.endswith(".pdf"):

                        doc = fitz.open(file_path)

                        for page in doc:
                            text += page.get_text()

                    elif file.endswith(".docx"):

                        doc = Document(file_path)

                        for para in doc.paragraphs:
                            text += para.text + " "

                    clean_text = re.sub(
                        r'\s+',
                        ' ',
                        text
                    )

                    start = 0

                    while start < len(clean_text):

                        end = start + chunk_size

                        chunk = clean_text[start:end]

                        all_chunks.append({

                            "source": file,

                            "text": chunk
                        })

                        start += (
                            chunk_size - chunk_overlap
                        )

                except Exception as e:

                    st.error(
                        f"Error processing {file}: {e}"
                    )

            # -----------------------------------
            # EMBEDDINGS
            # -----------------------------------

            chunk_texts = [

                chunk["text"]

                for chunk in all_chunks
            ]

            embeddings = embedding_model.encode(
                chunk_texts
            )

            embeddings = np.array(
                embeddings
            ).astype("float32")

            dimension = embeddings.shape[1]

            index = faiss.IndexFlatL2(
                dimension
            )

            index.add(embeddings)

            query_embedding = embedding_model.encode(
                [requirement]
            )

            query_embedding = np.array(
                query_embedding
            ).astype("float32")

            distances, indices = index.search(
                query_embedding,
                5
            )

            # -----------------------------------
            # CONTEXT
            # -----------------------------------

            final_context = ""

            reference_files = set()

            st.subheader(
                "Relevant Repository Matches"
            )

            for idx in indices[0]:

                match = all_chunks[idx]

                reference_files.add(
                    match["source"]
                )

                final_context += (
                    f"\nSource: {match['source']}\n"
                )

                final_context += (
                    match["text"] + "\n"
                )

                st.markdown("---")

                st.write(match["text"])

            final_context = final_context[:12000]

            # -----------------------------------
            # PROMPT
            # -----------------------------------

            prompt = f"""
You are an ERP Product Management AI Assistant.

Enterprise Repository Context:
{final_context}

Requirement:
{requirement}

Generate structured enterprise-grade FSD.

Return:

1. Objective
2. Scope
3. Functional Requirements
4. Workflow
5. Business Rules
6. Required Validations
7. Dependencies
8. Risks
9. Acceptance Criteria
10. Impacted Modules
11. Regression Areas
"""

            # -----------------------------------
            # API
            # -----------------------------------

            api_key = st.secrets["GEMINI_API_KEY"]

            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1/models/gemini-2.5-flash:generateContent?key={api_key}"
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                url,
                json=payload
            )

            response_json = response.json()

            try:

                ai_output = response_json[
                    "candidates"
                ][0]["content"]["parts"][0]["text"]

                st.session_state.generated_fsd = ai_output

                st.subheader("Generated FSD")

                st.write(ai_output)

                st.subheader(
                    "Reference Documents Used"
                )

                for ref in reference_files:
                    st.write(f"- {ref}")

            except:

                st.error(
                    "AI generation failed"
                )

                st.write(response_json)

    # -----------------------------------
    # PM REVIEW
    # -----------------------------------

    if st.session_state.generated_fsd:

        st.subheader("PM Review & Edit")

        edited_fsd = st.text_area(
            "Review / Edit Generated FSD",
            value=st.session_state.generated_fsd,
            height=500
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button("Approve FSD"):

                st.session_state.approved_fsd = edited_fsd

                st.success(
                    "FSD Approved Successfully"
                )

        with col2:

            section = st.selectbox(
                "Regenerate Section",
                [
                    "Business Rules",
                    "Required Validations",
                    "Acceptance Criteria",
                    "Regression Areas"
                ]
            )

        if st.button("Regenerate Selected Section"):

            regenerate_prompt = f"""
Requirement:
{requirement}

Existing FSD:
{edited_fsd}

Regenerate only:
{section}
"""

            api_key = st.secrets["GEMINI_API_KEY"]

            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1/models/gemini-2.5-flash:generateContent?key={api_key}"
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": regenerate_prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                url,
                json=payload
            )

            response_json = response.json()

            try:

                regenerated_output = response_json[
                    "candidates"
                ][0]["content"]["parts"][0]["text"]

                st.subheader(
                    f"Regenerated {section}"
                )

                st.write(regenerated_output)

            except:

                st.error(
                    "Section regeneration failed"
                )

    # -----------------------------------
    # EXPORT
    # -----------------------------------

    if st.session_state.approved_fsd:

        approved_fsd = st.session_state.approved_fsd

        st.subheader("Approved FSD")

        st.write(approved_fsd)

        if st.button("Export Approved FSD"):

            doc = Document()

            doc.add_heading(
                "Functional Specification Document",
                level=1
            )

            current_date = datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            )

            doc.add_paragraph(
                f"Generated On: {current_date}"
            )

            doc.add_paragraph(
                "Generated By: AI PM Assistant"
            )

            doc.add_paragraph(
                approved_fsd
            )

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            file_name = (
                f"generated_fsds/FSD_{timestamp}.docx"
            )

            doc.save(file_name)

            st.success(
                "FSD Exported Successfully"
            )

            with open(file_name, "rb") as file:

                st.download_button(
                    label="Download FSD",
                    data=file,
                    file_name=f"FSD_{timestamp}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

# ===================================
# TAB 2 - QA
# ===================================

with tab2:

    st.subheader(
        "QA Test Case Generation"
    )

    if st.session_state.approved_fsd:

        if st.button("Generate Test Cases"):

            approved_fsd = st.session_state.approved_fsd

            test_case_prompt = f"""
You are a QA Test Engineer.

Approved FSD:
{approved_fsd}

Generate:

1. Functional Test Cases
2. Validation Test Cases
3. Negative Test Cases
4. Regression Test Areas
5. Edge Cases
"""

            api_key = st.secrets["GEMINI_API_KEY"]

            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1/models/gemini-2.5-flash:generateContent?key={api_key}"
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": test_case_prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                url,
                json=payload
            )

            response_json = response.json()

            try:

                test_cases = response_json[
                    "candidates"
                ][0]["content"]["parts"][0]["text"]

                st.write(test_cases)

            except:

                st.error(
                    "Test case generation failed"
                )

    else:

        st.warning(
            "Approve FSD before generating QA test cases"
        )

# ===================================
# TAB 3 - RELEASE NOTES
# ===================================

with tab3:

    st.subheader("Release Notes")

    if st.session_state.approved_fsd:

        if st.button(
            "Generate Release Notes"
        ):

            release_prompt = f"""
Generate enterprise release notes from this FSD.

FSD:
{st.session_state.approved_fsd}

Return:

1. Feature Summary
2. Key Changes
3. Business Impact
4. Important Notes
5. User Actions Required
"""

            api_key = st.secrets["GEMINI_API_KEY"]

            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1/models/gemini-2.5-flash:generateContent?key={api_key}"
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": release_prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                url,
                json=payload
            )

            response_json = response.json()

            try:

                release_notes = response_json[
                    "candidates"
                ][0]["content"]["parts"][0]["text"]

                st.write(release_notes)

            except:

                st.error(
                    "Release note generation failed"
                )

    else:

        st.warning(
            "Approve FSD before generating release notes"
        )
