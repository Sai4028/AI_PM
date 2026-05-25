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
from docx.shared import Inches

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI PM Assistant",
    layout="wide"
)

# -----------------------------------
#  ERP UI THEME
# -----------------------------------

st.markdown(
    """
    <style>

    /* GLOBAL */

    .stApp {

        background-color: #F5F7FB;

        color: #1E293B;

        font-family: "Segoe UI", sans-serif;
    }

    /* REMOVE DEFAULT TOP SPACE */

    .block-container {

        padding-top: 2rem;

        padding-left: 2rem;

        padding-right: 2rem;

        padding-bottom: 2rem;
    }

    /* TITLES */

    h1 {

        color: #1E293B;

        font-size: 2.5rem;

        font-weight: 700;
    }

    h2 {

        color: #334155;

        font-weight: 600;
    }

    h3 {

        color: #475569;

        font-weight: 600;
    }

    /* CARDS */

    div[data-testid="stVerticalBlock"] > div {

        background-color: white;

        border-radius: 18px;

        padding: 1.2rem;

        border: 1px solid #E2E8F0;

        box-shadow: 0px 2px 10px rgba(0,0,0,0.04);

        margin-bottom: 1rem;
    }

    /* BUTTONS */

    .stButton > button {

        width: 100%;

        background: linear-gradient(
            90deg,
            #34D399,
            #10B981
        );

        color: white;

        border: none;

        border-radius: 12px;

        padding: 0.7rem 1rem;

        font-size: 15px;

        font-weight: 600;

        transition: 0.3s;
    }

    .stButton > button:hover {

        background: linear-gradient(
            90deg,
            #10B981,
            #059669
        );

        color: white;

        transform: scale(1.01);
    }

    /* DOWNLOAD BUTTON */

    .stDownloadButton > button {

        width: 100%;

        background: linear-gradient(
            90deg,
            #6366F1,
            #4F46E5
        );

        color: white;

        border: none;

        border-radius: 12px;

        padding: 0.7rem 1rem;

        font-size: 15px;

        font-weight: 600;
    }

    .stDownloadButton > button:hover {

        background: linear-gradient(
            90deg,
            #4F46E5,
            #4338CA
        );

        color: white;
    }

    /* TEXT AREA */

    .stTextArea textarea {

        border-radius: 14px;

        border: 1px solid #CBD5E1;

        background-color: #FFFFFF;

        color: #111827;

        padding: 1rem;

        font-size: 15px;
    }

    /* INPUT */

    .stTextInput input {

        border-radius: 12px;

        border: 1px solid #CBD5E1;
    }

    /* FILE UPLOADER */

    section[data-testid="stFileUploader"] {

        border-radius: 14px;

        border: 1px dashed #94A3B8;

        background-color: #FFFFFF;

        padding: 1rem;
    }

    /* SELECTBOX */

    .stSelectbox div[data-baseweb="select"] {

        border-radius: 12px;
    }

    /* EXPANDER */

    .streamlit-expanderHeader {

        background-color: #ECFDF5;

        border-radius: 10px;

        color: #065F46;

        font-weight: 600;
    }

    /* SUCCESS */

    .stSuccess {

        background-color: #DCFCE7;

        color: #166534;

        border-radius: 12px;

        padding: 1rem;
    }

    /* WARNING */

    .stWarning {

        background-color: #FEF3C7;

        color: #92400E;

        border-radius: 12px;

        padding: 1rem;
    }

    /* ERROR */

    .stError {

        background-color: #FEE2E2;

        color: #991B1B;

        border-radius: 12px;

        padding: 1rem;
    }

    /* HORIZONTAL LINE */

    hr {

        border-top: 1px solid #E2E8F0;
    }

    /* SCROLLBAR */

    ::-webkit-scrollbar {

        width: 8px;
    }

    ::-webkit-scrollbar-thumb {

        background: #CBD5E1;

        border-radius: 10px;
    }

    /* METRIC STYLE */

    div[data-testid="metric-container"] {

        background-color: white;

        border-radius: 16px;

        border: 1px solid #E2E8F0;

        padding: 1rem;

        box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
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
# UI TITLE
# -----------------------------------

st.title("AI PM Assistant")

# -----------------------------------
# REPOSITORY STATUS
# -----------------------------------

repository_files = os.listdir("repository")

st.subheader(
    f"Repository Files Count: {len(repository_files)}"
)

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
# REQUIREMENT INPUT
# -----------------------------------

st.subheader("Enter Requirement")

requirement = st.text_area(
    "Requirement",
    height=200
)

analyze = st.button("Generate FSD")

# -----------------------------------
# MAIN FLOW
# -----------------------------------

if analyze:

    if not requirement:

        st.error("Please enter requirement")

    else:

        # -----------------------------------
        # SAVE FILES
        # -----------------------------------

        if uploaded_files:

            for uploaded_file in uploaded_files:

                file_path = os.path.join(
                    "repository",
                    uploaded_file.name
                )

                with open(file_path, "wb") as f:

                    f.write(uploaded_file.getbuffer())

        # -----------------------------------
        # READ DOCS
        # -----------------------------------

        repository_files = os.listdir("repository")

        all_chunks = []

        chunk_size = 800
        chunk_overlap = 100

        for file in repository_files:

            file_path = os.path.join(
                "repository",
                file
            )

            text = ""

            try:

                # PDF

                if file.endswith(".pdf"):

                    doc = fitz.open(file_path)

                    for page in doc:

                        text += page.get_text()

                # DOCX

                elif file.endswith(".docx"):

                    doc = Document(file_path)

                    for para in doc.paragraphs:

                        text += para.text + " "

                # CLEAN

                clean_text = re.sub(
                    r'\s+',
                    ' ',
                    text
                )

                # CHUNKING

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

                st.error(f"Error processing {file}: {e}")

        # -----------------------------------
        # EMPTY CHECK
        # -----------------------------------

        if not all_chunks:

            st.warning(
                "No repository documents found"
            )

        else:

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

            # -----------------------------------
            # FAISS
            # -----------------------------------

            dimension = embeddings.shape[1]

            index = faiss.IndexFlatL2(
                dimension
            )

            index.add(embeddings)

            # -----------------------------------
            # QUERY
            # -----------------------------------

            query_embedding = embedding_model.encode(
                [requirement]
            )

            query_embedding = np.array(
                query_embedding
            ).astype("float32")

            # -----------------------------------
            # SEARCH
            # -----------------------------------

            k = 5

            distances, indices = index.search(
                query_embedding,
                k
            )

            # -----------------------------------
            # BUILD CONTEXT
            # -----------------------------------

            final_context = ""

            reference_files = set()

            st.subheader(
                "Relevant Repository Matches"
            )

            rank = 1

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

                st.markdown(
                    f"### Match Rank: {rank}"
                )

                st.write(match["text"])

                rank += 1

            # LIMIT CONTEXT

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

Generate a structured enterprise-grade Functional Specification Document.

Return output in the following format:

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
11. Regression Testing Areas

Avoid unsupported assumptions.
Use repository references where relevant.
"""

            # -----------------------------------
            # GEMINI API
            # -----------------------------------

            api_key = st.secrets["GEMINI_API_KEY"]

            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1/models/gemini-2.5-flash:generateContent?key={api_key}"
            )

            headers = {
                "Content-Type": "application/json"
            }

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
                headers=headers,
                json=payload
            )

            response_json = response.json()

            # -----------------------------------
            # RESPONSE
            # -----------------------------------

            try:

                ai_output = response_json[
                    "candidates"
                ][0]["content"]["parts"][0]["text"]

                st.session_state.generated_fsd = ai_output

                st.subheader(
                    "Generated FSD"
                )

                st.write(ai_output)

                # -----------------------------------
                # REFERENCES USED
                # -----------------------------------

                st.subheader(
                    "Reference Documents Used"
                )

                for ref in reference_files:

                    st.write(f"- {ref}")

            except Exception as e:

                st.error(
                    "AI generation failed"
                )

                st.write(response_json)

# -----------------------------------
# PM REVIEW SECTION
# -----------------------------------

if st.session_state.generated_fsd:

    st.subheader("PM Review & Edit")

    edited_fsd = st.text_area(

        "Review / Edit Generated FSD",

        value=st.session_state.generated_fsd,

        height=500
    )

    # -----------------------------------
    # APPROVE BUTTON
    # -----------------------------------

    if st.button("Approve FSD"):

        st.session_state.approved_fsd = edited_fsd

        st.success(
            "FSD Approved Successfully"
        )

    # -----------------------------------
    # SECTION REGENERATE
    # -----------------------------------

    st.subheader(
        "Regenerate Specific Section"
    )

    section = st.selectbox(

        "Select Section",

        [
            "Business Rules",
            "Required Validations",
            "Acceptance Criteria",
            "Regression Testing Areas"
        ]
    )

    if st.button("Regenerate Section"):

        regenerate_prompt = f"""
Requirement:
{requirement}

Existing FSD:
{edited_fsd}

Regenerate only this section:
{section}

Return only the regenerated section.
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
# SAVE APPROVED FSD
# -----------------------------------

if st.session_state.approved_fsd:

    approved_fsd = st.session_state.approved_fsd

    st.subheader(
        "Approved FSD"
    )

    st.write(approved_fsd)

    # -----------------------------------
    # SAVE DOCX
    # -----------------------------------

    if st.button("Export Approved FSD"):

        doc = Document()

        # TITLE

        doc.add_heading(
            "Functional Specification Document",
            level=1
        )

        # DATE

        current_date = datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        )

        doc.add_paragraph(
            f"Generated On: {current_date}"
        )

        doc.add_paragraph(
            "Generated By: AI PM Assistant"
        )

        # CONTENT

        doc.add_paragraph(
            approved_fsd
        )

        # FILE NAME

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

        # DOWNLOAD BUTTON

        with open(file_name, "rb") as file:

            st.download_button(

                label="Download FSD",

                data=file,

                file_name=f"FSD_{timestamp}.docx",

                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    # -----------------------------------
    # QA TEST CASE GENERATION
    # -----------------------------------

    st.subheader(
        "Generate QA Test Cases"
    )

    if st.button(
        "Generate Test Cases"
    ):

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

Provide structured output.
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

            st.subheader(
                "Generated QA Test Cases"
            )

            st.write(test_cases)

        except:

            st.error(
                "Test case generation failed"
            )
