
import streamlit as st
import fitz
import re

st.set_page_config(page_title="AI PM Assistant", layout="wide")

st.title("AI PM Assistant")

st.subheader("Upload Enterprise Documents")

uploaded_files = st.file_uploader(
    "Upload FSDs, PPTs, APIs, Reports",
    accept_multiple_files=True,
    type=["pdf"]
)

st.subheader("Enter New Requirement")

requirement = st.text_area(
    "Requirement",
    height=150
)

analyze = st.button("Analyze Requirement")

if analyze:

    if not uploaded_files:
        st.error("Please upload documents")
    
    elif not requirement:
        st.error("Please enter requirement")

    else:

        st.success("Analysis Started")

        # STEP 1 - Extract Text
        all_text = ""

        for uploaded_file in uploaded_files:

            pdf_bytes = uploaded_file.read()

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            for page in doc:
                all_text += page.get_text()

        # STEP 2 - Requirement Understanding
        requirement_lower = requirement.lower()

        generated_keywords = []

        if "gst" in requirement_lower:
            generated_keywords.extend(["gst", "tax"])

        if "registration" in requirement_lower:
            generated_keywords.extend(["registration", "register", "onboarding"])

        if "supplier" in requirement_lower:
            generated_keywords.extend(["supplier", "vendor"])

        if "validation" in requirement_lower:
            generated_keywords.extend(["validation", "mandatory", "verify"])

        # STEP 3 - Clean Text
        clean_text = re.sub(r'\s+', ' ', all_text)

        # STEP 4 - Chunking
        chunk_size = 700
        chunks = []

        for i in range(0, len(clean_text), chunk_size):
            chunks.append(clean_text[i:i+chunk_size])

        # STEP 5 - Retrieval
        matches = []

        for chunk in chunks:

            score = 0

            for keyword in generated_keywords:

                if keyword.lower() in chunk.lower():
                    score += 1

            if score >= 2:
                matches.append((score, chunk))

        matches.sort(reverse=True, key=lambda x: x[0])

        # STEP 6 - Context Building
        final_context = ""

        for score, match in matches[:5]:
            final_context += match + "\n\n"

        # STEP 7 - Simulated AI Output

        st.subheader("Requirement Summary")

        st.write(requirement)

        st.subheader("Generated Keywords")

        st.write(generated_keywords)

        st.subheader("Relevant Enterprise Context")

        st.write(final_context)

        st.subheader("Impact Analysis")

        st.write("""
        - Supplier Registration workflow impacted
        - Validation engine impacted
        - Supplier onboarding impacted
        - Approval workflow may be impacted
        - Notification flow may be impacted
        """)

        st.subheader("Suggested Regression Areas")

        st.write("""
        - Supplier registration
        - Passcode validation
        - Duplicate registration
        - Approval flow
        - Email notifications
        """)
