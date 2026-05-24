import streamlit as st
import fitz
import os
import re
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

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

st.subheader("Upload Additional Documents")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

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
        # SAVE UPLOADED FILES
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
        # READ ALL REPOSITORY FILES
        # -----------------------------------

        all_chunks = []

        repository_files = os.listdir(
            "repository"
        )

        for file in repository_files:

            if file.endswith(".pdf"):

                file_path = os.path.join(
                    "repository",
                    file
                )

                doc = fitz.open(file_path)

                text = ""

                for page in doc:

                    text += page.get_text()

                # -----------------------------------
                # CLEAN TEXT
                # -----------------------------------

                clean_text = re.sub(
                    r'\\s+',
                    ' ',
                    text
                )

                # -----------------------------------
                # CHUNKING
                # -----------------------------------

                chunk_size = 400

                for i in range(
                    0,
                    len(clean_text),
                    chunk_size
                ):

                    chunk = clean_text[i:i+chunk_size]

                    all_chunks.append({
                        "source": file,
                        "text": chunk
                    })

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
        # DISPLAY RESULTS
        # -----------------------------------

        st.subheader(
            "Relevant Repository Matches"
        )

        for idx in indices[0]:

            match = all_chunks[idx]

            st.markdown("---")

            st.markdown(
                f"### Source: {match['source']}"
            )

            st.write(match["text"])
