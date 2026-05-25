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
# CLEAN ENTERPRISE UI
# -----------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F8FAFC;
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
        color: #0F172A;
        font-size: 2.8rem;
        font-weight: 700;
    }

    h2, h3 {
        color: #1E293B;
        font-weight: 600;
    }

    div[data-testid="metric-container"] {

        background: white;

        border: 1px solid #E2E8F0;

        padding: 1rem;

        border-radius: 12px;
    }

    .stButton > button {

        background-color: #2563EB;

        color: white;

        border: none;

        border-radius: 8px;

        padding: 0.6rem 1rem;

        font-weight: 600;
    }

    .stButton > button:hover {

        background-color: #1D4ED8;

        color: white;
    }

    .stDownloadButton > button {

        background-color: #059669;

        color: white;

        border: none;

        border-radius: 8px;

        padding: 0.6rem 1rem;

        font-weight: 600;
    }

    .stTextArea textarea {

        background-color: white;

        color: #111827;

        border: 1px solid #CBD5E1;

        border-radius: 10px;
    }

    section[data-testid="stFileUploader"] {

        background-color: white;

        border: 1px dashed #CBD5E1;

        border-radius: 10px;

        padding: 1rem;
    }

    button[data-baseweb="tab"] {

        color: #334155 !important;

        font-weight: 600;

        font-size: 15px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {

        color: #2563EB !important;

        border-bottom: 2px solid #2563EB;
    }

    .streamlit-expanderHeader {

        color: #1E293B;

        font-weight: 600;
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

if "qa_generated" not in st.session_state:
    st.session_state.qa_generated = False

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "FSD Generation",
    "QA Test Cases",
    "Release Notes",
    "Support Notes",
    "GTM Content",
    "PPT Generator"
])
# ===================================
# TAB 1
# ===================================

with tab1:

    with st.expander("View Repository Files"):

        if repository_files:

            for file in repository_files:
                st.write(file)

        else:
            st.warning("Repository is empty")

    st.subheader("Upload Additional Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    st.subheader("Enter Requirement")

    requirement = st.text_area(
        "Requirement",
        height=200
    )

    analyze = st.button("Generate FSD")

    if analyze:

        if not requirement:

            st.error("Please enter requirement")

        else:

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

    if st.session_state.generated_fsd:

        st.subheader("PM Review & Edit")

        edited_fsd = st.text_area(
            "Review / Edit Generated FSD",
            value=st.session_state.generated_fsd,
            height=500
        )

        if st.button("Approve FSD"):

            st.session_state.approved_fsd = edited_fsd

            st.success(
                "FSD Approved Successfully"
            )

        if st.session_state.approved_fsd:

            approved_fsd = st.session_state.approved_fsd

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
# TAB 2
# ===================================

with tab2:

    st.subheader(
        "QA Test Case Generation"
    )

    if st.session_state.approved_fsd:

        if st.button("Generate QA Test Cases"):

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

                st.session_state.qa_generated = True

                st.success(
                    "QA Test Cases Generated Successfully"
                )

                st.write(test_cases)

                qa_doc = Document()

                qa_doc.add_heading(
                    "QA Test Cases",
                    level=1
                )

                current_date = datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )

                qa_doc.add_paragraph(
                    f"Generated On: {current_date}"
                )

                qa_doc.add_paragraph(
                    "Generated By: AI PM Assistant"
                )

                qa_doc.add_paragraph(
                    test_cases
                )

                qa_file_name = (
                    f"generated_fsds/QA_Test_Cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                )

                qa_doc.save(qa_file_name)

                with open(qa_file_name, "rb") as file:

                    st.download_button(

                        label="Download QA Test Cases",

                        data=file,

                        file_name=os.path.basename(
                            qa_file_name
                        ),

                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

            except:

                st.error(
                    "QA generation failed"
                )

    else:

        st.warning(
            "Approve FSD before generating QA Test Cases"
        )

# ===================================
# TAB 3
# ===================================

with tab3:

    st.subheader("Release Notes")

    if st.session_state.qa_generated:

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

                release_doc = Document()

                release_doc.add_heading(
                    "Release Notes",
                    level=1
                )

                current_date = datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )

                release_doc.add_paragraph(
                    f"Generated On: {current_date}"
                )

                release_doc.add_paragraph(
                    "Generated By: AI PM Assistant"
                )

                release_doc.add_paragraph(
                    release_notes
                )

                release_file_name = (
                    f"generated_fsds/Release_Notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                )

                release_doc.save(release_file_name)

                with open(release_file_name, "rb") as file:

                    st.download_button(

                        label="Download Release Notes",

                        data=file,

                        file_name=os.path.basename(
                            release_file_name
                        ),

                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

            except:

                st.error(
                    "Release note generation failed"
                )

    else:

        st.warning(
            "Generate QA Test Cases before Release Notes"
        )
# ===================================
# TAB 4 - SUPPORT NOTES
# ===================================

with tab4:

    st.subheader("Support Notes")

    if st.session_state.qa_generated:

        if st.button(
            "Generate Support Notes"
        ):

            support_prompt = f"""
Generate enterprise support notes from this FSD.

FSD:
{st.session_state.approved_fsd}

Return:

1. Feature Overview
2. Configuration Dependencies
3. Known Limitations
4. Troubleshooting Notes
5. Common User Issues
6. Important Support Considerations
7. Impacted Areas
8. FAQs
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
                                "text": support_prompt
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

                support_notes = response_json[
                    "candidates"
                ][0]["content"]["parts"][0]["text"]

                st.write(support_notes)

                # -----------------------------------
                # EXPORT SUPPORT NOTES
                # -----------------------------------

                support_doc = Document()

                support_doc.add_heading(
                    "Support Notes",
                    level=1
                )

                current_date = datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )

                support_doc.add_paragraph(
                    f"Generated On: {current_date}"
                )

                support_doc.add_paragraph(
                    "Generated By: AI PM Assistant"
                )

                support_doc.add_paragraph(
                    support_notes
                )

                support_file_name = (
                    f"generated_fsds/Support_Notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                )

                support_doc.save(support_file_name)

                with open(support_file_name, "rb") as file:

                    st.download_button(

                        label="Download Support Notes",

                        data=file,

                        file_name=os.path.basename(
                            support_file_name
                        ),

                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

            except:

                st.error(
                    "Support note generation failed"
                )

    else:

        st.warning(
            "Generate QA Test Cases before Support Notes"
        )
# ===================================
# TAB 5 - GTM CONTENT
# ===================================

with tab5:

    st.subheader("GTM Content Generation")

    if st.session_state.qa_generated:

        if st.button(
            "Generate GTM Content"
        ):

            gtm_prompt = f"""
Generate GTM content from this FSD.

FSD:
{st.session_state.approved_fsd}

Return:

1. Feature Highlights
2. Business Benefits
3. Customer Value Proposition
4. Sales Talking Points
5. Marketing Summary
6. Brochure Content
7. PPT Summary
8. Customer Announcement Draft
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
                                "text": gtm_prompt
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

                gtm_content = response_json[
                    "candidates"
                ][0]["content"]["parts"][0]["text"]

                st.write(gtm_content)

                # -----------------------------------
                # EXPORT GTM CONTENT
                # -----------------------------------

                gtm_doc = Document()

                gtm_doc.add_heading(
                    "GTM Content",
                    level=1
                )

                current_date = datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )

                gtm_doc.add_paragraph(
                    f"Generated On: {current_date}"
                )

                gtm_doc.add_paragraph(
                    "Generated By: AI PM Assistant"
                )

                gtm_doc.add_paragraph(
                    gtm_content
                )

                gtm_file_name = (
                    f"generated_fsds/GTM_Content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                )

                gtm_doc.save(gtm_file_name)

                with open(gtm_file_name, "rb") as file:

                    st.download_button(

                        label="Download GTM Content",

                        data=file,

                        file_name=os.path.basename(
                            gtm_file_name
                        ),

                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

            except:

                st.error(
                    "GTM content generation failed"
                )

    else:

        st.warning(
            "Generate QA Test Cases before GTM Content"
        )

# ===================================
# TAB 6 - PPT GENERATOR
# ===================================

from pptx import Presentation
from pptx.util import Inches

with tab6:

    st.subheader("GTM PPT Generator")

    if st.session_state.qa_generated:

        if st.button(
            "Generate PPT"
        ):

            ppt_prompt = f"""
Generate presentation slide content from this FSD.

FSD:
{st.session_state.approved_fsd}

Return slide-wise content.

Format:

Slide 1: Title
Slide 2: Business Problem
Slide 3: Feature Overview
Slide 4: Key Functionalities
Slide 5: Business Benefits
Slide 6: Workflow Summary
Slide 7: Customer Value
Slide 8: Conclusion
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
                                "text": ppt_prompt
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

                ppt_content = response_json[
                    "candidates"
                ][0]["content"]["parts"][0]["text"]

                st.write(ppt_content)

                # -----------------------------------
                # CREATE PPT
                # -----------------------------------

                prs = Presentation()

                slides_data = ppt_content.split("Slide ")

                for slide_data in slides_data:

                    if slide_data.strip() == "":
                        continue

                    lines = slide_data.split("\n")

                    title = lines[0]

                    content = "\n".join(lines[1:])

                    slide_layout = prs.slide_layouts[1]

                    slide = prs.slides.add_slide(
                        slide_layout
                    )

                    slide.shapes.title.text = title

                    slide.placeholders[1].text = content

                ppt_file_name = (
                    f"generated_fsds/GTM_Presentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
                )

                prs.save(ppt_file_name)

                st.success(
                    "PPT Generated Successfully"
                )

                with open(ppt_file_name, "rb") as file:

                    st.download_button(

                        label="Download PPT",

                        data=file,

                        file_name=os.path.basename(
                            ppt_file_name
                        ),

                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )

            except:

                st.error(
                    "PPT generation failed"
                )

    else:

        st.warning(
            "Generate QA Test Cases before PPT Generation"
        )
