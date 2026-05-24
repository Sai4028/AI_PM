import streamlit as st
import fitz
import os
import re
import faiss
import numpy as np
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from docx import Document

# -----------------------------------
# GEMINI CONFIG
# -----------------------------------

genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI PM Assistant",
    layout="wide"
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
# UI
# -----------------------------------

st.title("AI PM Assistant")

# -----------------------------------
# REPOSITORY FILES
# -----------------------------------

st.subheader("Repository Files")

repository_files = os.listdir("repository")

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
# REQUIREMENT INPUT
# -----------------------------------

st.subheader("Enter Requirement")

requirement = st.text_area(
    "Requirement",
    height=150
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

        # -----------------------------------
        # FILTER FILES
        # -----------------------------------

        repository_files = os.listdir(
            "repository"
        )

        filtered_files = []

        requirement_lower = requirement.lower()

        for file in repository_files:

            file_lower = file.lower()

            if "customer" in requirement_lower:

                if "customer" in file_lower:

                    filtered_files.append(file)

            elif (
                "supplier" in requirement_lower
                or
                "vendor" in requirement_lower
            ):

                if (
                    "supplier" in file_lower
                    or
                    "vendor" in file_lower
                ):

                    filtered_files.append(file)

            elif "inventory" in requirement_lower:

                if "inventory" in file_lower:

                    filtered_files.append(file)

            elif "finance" in requirement_lower:

                if "finance" in file_lower:

                    filtered_files.append(file)

            else:

                filtered_files.append(file)

        # -----------------------------------
        # READ DOCUMENTS
        # -----------------------------------

        all_chunks = []

        for file in filtered_files:

            file_path = os.path.join(
                "repository",
                file
            )

            text = ""

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

            chunk_size = 400

            for i in range(
                0,
                len(clean_text),
                chunk_size
            ):

                chunk = clean_text[
                    i:i+chunk_size
                ]

                all_chunks.append({

                    "source": file,

                    "text": chunk

                })

        # -----------------------------------
        # CHECK EMPTY
        # -----------------------------------

        if not all_chunks:

            st.warning(
                "No matching repository documents found"
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
            # FAISS INDEX
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

            for idx in indices[0]:

                match = all_chunks[idx]

                final_context += (
                    f"Source: {match['source']}\n"
                )

                final_context += (
                    match["text"] + "\n\n"
                )

            # -----------------------------------
            # DISPLAY MATCHES
            # -----------------------------------

            st.subheader(
                "Relevant Repository Matches"
            )

            rank = 1

            for idx in indices[0]:

                match = all_chunks[idx]

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
            # AI REASONING
            # -----------------------------------

            prompt = f"""
You are an ERP Product Management AI Assistant.

Enterprise Context:
{final_context}

Requirement:
{requirement}

Analyze and provide:

1. Requirement Summary
2. Impacted Modules
3. Required Validations
4. Regression Testing Areas
5. Risks and Dependencies

Provide structured output.
"""

            response = model.generate_content(
                prompt
            )

            ai_output = response.text

            # -----------------------------------
            # DISPLAY AI ANALYSIS
            # -----------------------------------

            st.subheader(
                "AI Analysis"
            )

            st.write(ai_output)
