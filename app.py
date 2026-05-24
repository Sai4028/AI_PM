
import streamlit as st
import fitz
import os
import re
import chromadb

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
# CHROMA DB
# -----------------------------------

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="enterprise_docs"
)

# -----------------------------------
# UI
# -----------------------------------

st.title("AI PM Assistant")

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

        all_text = ""

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

                for page in doc:

                    all_text += page.get_text()

        # -----------------------------------
        # CLEAN TEXT
        # -----------------------------------

        clean_text = re.sub(
            r'\\s+',
            ' ',
            all_text
        )

        # -----------------------------------
        # CHUNKING
        # -----------------------------------

        chunk_size = 400

        chunks = []

        for i in range(
            0,
            len(clean_text),
            chunk_size
        ):

            chunks.append(
                clean_text[i:i+chunk_size]
            )

        # -----------------------------------
        # RESET VECTOR DB
        # -----------------------------------

        try:

            collection.delete(where={})

        except:

            pass

        # -----------------------------------
        # STORE EMBEDDINGS
        # -----------------------------------

        for idx, chunk in enumerate(chunks):

            embedding = embedding_model.encode(
                chunk
            ).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(idx)]
            )

        # -----------------------------------
        # SEMANTIC SEARCH
        # -----------------------------------

        query_embedding = embedding_model.encode(
            requirement
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        retrieved_chunks = results["documents"][0]

        # -----------------------------------
        # DISPLAY RESULTS
        # -----------------------------------

        st.subheader(
            "Relevant Repository Matches"
        )

        if retrieved_chunks:

            for match in retrieved_chunks:

                st.markdown("---")

                st.write(match)

        else:

            st.warning(
                "No relevant matches found"
            )
