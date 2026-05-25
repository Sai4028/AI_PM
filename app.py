import streamlit as st
import fitz
import os
import re
import faiss
import numpy as np
import requests

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
# ERP UI THEME
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
# CREATE REPOSITORY
# -----------------------------------

os.makedirs("repository", exist_ok=True)

# -----------------------------------
# EMBEDDING MODEL
# -----------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
# UI TITLE
# -----------------------------------

st.title("AI PM Assistant")

# -----------------------------------
# REPOSITORY STATUS
# -----------------------------------

repository_files = os.listdir("repository")

repo_count = len(repository_files)

st.subheader(
    f"Repository Files Count: {repo_count}"
)

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

uploaded_count = 0

if uploaded_files:

    uploaded_count = len(uploaded_files)

st.subheader(
    f"Uploaded Files Count: {uploaded_count}"
)

with st.expander("View Uploaded Files"):

    if uploaded_files:

        for file in uploaded_files:

            st.write(file.name)

    else:

        st.write("No uploaded files")

# -----------------------------------
# REQUIREMENT INPUT
# -----------------------------------

st.subheader("Enter Requirement")

requirement = st.text_area(
    "Requirement",
    height=200
)

analyze = st.button("Analyze Requirement")

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

                    f.write(
                        uploaded_file.getbuffer()
                    )

                st.success(
                    f"{uploaded_file.name} saved successfully"
                )

        # -----------------------------------
        # READ DOCUMENTS
        # -----------------------------------

        repository_files = os.listdir(
            "repository"
        )

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

                # -----------------------------------
                # CLEAN TEXT
                # -----------------------------------

                clean_text = re.sub(
                    r'\s+',
                    ' ',
                    text
                )

                # -----------------------------------
                # CHUNKING
                # -----------------------------------

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
        # CHECK EMPTY CHUNKS
        # -----------------------------------

        if not all_chunks:

            st.warning(
                "No repository content found"
            )

        else:

            # -----------------------------------
            # CREATE EMBEDDINGS
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
            # CREATE FAISS INDEX
            # -----------------------------------

            dimension = embeddings.shape[1]

            index = faiss.IndexFlatL2(
                dimension
            )

            index.add(embeddings)

            # -----------------------------------
            # QUERY EMBEDDING
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

            st.subheader(
                "Relevant Repository Matches"
            )

            rank = 1

            for idx in indices[0]:

                match = all_chunks[idx]

                final_context += (
                    f"\nSource: {match['source']}\n"
                )

                final_context += (
                    match["text"] + "\n"
                )

                st.markdown("---")

                st.markdown(
                    f"## Match Rank: {rank}"
                )

                st.markdown(
                    f"### Source File: {match['source']}"
                )

                st.write(match["text"])

                rank += 1

            # -----------------------------------
            # LIMIT CONTEXT
            # -----------------------------------

            final_context = final_context[:12000]

            # -----------------------------------
            # AI PROMPT
            # -----------------------------------

            prompt = f"""
You are an ERP Product Management AI Assistant.

Use the enterprise repository references provided below.

Enterprise Context:
{final_context}

Requirement:
{requirement}

Generate structured enterprise-grade analysis.

Return output with the following sections:

1. Requirement Summary
2. Functional Overview
3. Impacted Modules
4. Business Rules
5. Required Validations
6. Dependencies
7. Risks
8. Regression Testing Areas
9. Acceptance Criteria

Use repository references where relevant.
Avoid unsupported assumptions.
"""

            # -----------------------------------
            # GEMINI API CALL
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
            # SAFE RESPONSE HANDLING
            # -----------------------------------

            try:

                ai_output = response_json[
                    "candidates"
                ][0]["content"]["parts"][0]["text"]

                st.subheader(
                    "AI Analysis"
                )

                st.write(ai_output)

            except Exception as e:

                st.error(
                    "AI response generation failed"
                )

                st.write(response_json)
