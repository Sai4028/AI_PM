
import streamlit as st
import fitz
import os
import re

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
                    f.write(uploaded_file.getbuffer())

        # -----------------------------------
        # READ ALL REPOSITORY FILES
        # -----------------------------------

        all_text = ""

        repository_files = os.listdir("repository")

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
        # REQUIREMENT UNDERSTANDING
        # -----------------------------------

        requirement_lower = requirement.lower()

        generated_keywords = requirement_lower.split()

        # -----------------------------------
        # CLEAN TEXT
        # -----------------------------------

        clean_text = re.sub(r'\s+', ' ', all_text)

        # -----------------------------------
        # CHUNKING
        # -----------------------------------

        chunk_size = 700

        chunks = []

        for i in range(0, len(clean_text), chunk_size):

            chunks.append(
                clean_text[i:i+chunk_size]
            )

        # -----------------------------------
        # RETRIEVAL
        # -----------------------------------

        matches = []

        for chunk in chunks:

            score = 0

            for keyword in generated_keywords:

                if keyword.lower() in chunk.lower():
                    score += 1

            if score >= 2:

                matches.append(
                    (score, chunk)
                )

        matches.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        # -----------------------------------
        # DISPLAY RESULTS
        # -----------------------------------

        st.subheader("Relevant Repository Matches")

        if matches:

            for score, match in matches[:5]:

                st.markdown("---")

                st.write(match)

        else:

            st.warning(
                "No relevant matches found"
            )
