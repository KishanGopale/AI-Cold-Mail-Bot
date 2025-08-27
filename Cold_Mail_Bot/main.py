# app.py
import os
import io
import requests
from bs4 import BeautifulSoup
import streamlit as st
from PyPDF2 import PdfReader

# LangChain + Groq + vectorstore imports
from langchain_groq import ChatGroq
from langchain import LLMChain, PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# ---------- Utilities ----------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

def extract_text_from_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # remove scripts & styles
        for s in soup(["script", "style", "header", "footer", "nav", "form"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        # basic cleaning: collapse repeated newlines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)
    except Exception as e:
        st.error(f"Failed to fetch URL: {e}")
        return ""

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Cold Mail Bot (LLM + Vector DB)")
st.title("Cold Mail Bot — Generate from JD + Resume")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Inputs")
    st.write("Provide **either** a Job Description (URL or PDF) but not both, and upload your Resume PDF below.")

    jd_input_type = st.radio("Job description input type", ("URL", "Upload PDF"))

    job_text = ""
    if jd_input_type == "URL":
        jd_url = st.text_input("Job description URL")
        jd_file = None
        if jd_url:
            if st.button("Fetch JD from URL"):
                job_text = extract_text_from_url(jd_url)
                if job_text:
                    st.success("Job description fetched.")
                    st.text_area("Job description (preview)", job_text[:5000], height=200)
    else:
        jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"])
        jd_url = None
        if jd_file is not None:
            if st.button("Read JD PDF"):
                job_text = extract_text_from_pdf_bytes(jd_file.read())
                if job_text:
                    st.success("Job description PDF read.")
                    st.text_area("Job description (preview)", job_text[:5000], height=200)

    st.markdown("---")
    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
    resume_text = ""
    if resume_file is not None:
        if st.button("Read Resume PDF"):
            resume_text = extract_text_from_pdf_bytes(resume_file.read())
            if resume_text:
                st.success("Resume read.")
                st.text_area("Resume (preview)", resume_text[:5000], height=200)

    st.markdown("---")
    st.caption("Options")
    top_k = st.slider("Retriever: top k documents to fetch", min_value=1, max_value=10, value=4)
    submit_btn = st.button("Generate Cold Email")

with right_col:
    st.subheader("Output")
    output_area = st.empty()

# ---------- Validation ----------
if submit_btn:
    # Validate edge condition: JD must be provided either via URL or PDF but not both; and resume required
    if (not job_text) and (not (jd_url or jd_file)):
        st.error("Please provide a job description (URL or PDF).")
    elif resume_file is None and not resume_text:
        st.error("Please upload and read a resume PDF.")
    else:
        # Ensure we have job_text (maybe fetched already). If a URL was entered but not fetched, fetch now.
        if not job_text:
            if jd_url:
                job_text = extract_text_from_url(jd_url)
            elif jd_file:
                jd_file.seek(0)
                job_text = extract_text_from_pdf_bytes(jd_file.read())

        if not resume_text and resume_file:
            resume_file.seek(0)
            resume_text = extract_text_from_pdf_bytes(resume_file.read())

        if not job_text or not resume_text:
            st.error("Couldn't extract text from the provided files/URL. Check files and try again.")
        else:
            with st.spinner("Indexing documents into vector DB and generating email..."):
                # ---------- Embeddings + Vectorstore ----------
                # Using sentence-transformers via langchain's HuggingFaceEmbeddings wrapper
                # This model is small and fast; change if you prefer a different embedding model.
                embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
                embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

                docs = []
                docs.append(Document(page_content=job_text, metadata={"type": "job_description"}))
                docs.append(Document(page_content=resume_text, metadata={"type": "resume"}))

                # Create FAISS index (in-memory). For production persist to disk or use a hosted vector DB.
                try:
                    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
                except Exception as e:
                    st.error(f"Failed to create vector DB: {e}")
                    raise

                retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

                # Get relevant docs for the job_text (this will return both JD & resume snippets)
                relevant_docs = retriever.get_relevant_documents(job_text)

                # Prepare context snippets (concatenate small excerpts)
                def snippet(d: Document, max_chars=1200):
                    txt = d.page_content
                    return txt[:max_chars].strip()

                jd_snips = []
                resume_snips = []
                for d in relevant_docs:
                    t = d.metadata.get("type", "")
                    if t == "job_description":
                        jd_snips.append(snippet(d))
                    elif t == "resume":
                        resume_snips.append(snippet(d))
                    else:
                        # fallback: include anywhere
                        resume_snips.append(snippet(d))

                # ---------- LLM (ChatGroq) ----------
                groq_key = os.getenv("GROQ_API_KEY")
                if not groq_key:
                    st.error("GROQ_API_KEY not found in environment variables. Set it and try again.")
                else:
                    llm = ChatGroq(
                        groq_api_key=groq_key,
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        temperature=0
                    )

                    # Construct prompt template
                    prompt_template = """
You are an assistant that writes a professional job inquiry email.
The candidate is looking for a job and wants to reach out to recruiters/hiring managers.

Write an email using:
1) Job Description (relevant requirements)
2) Candidate Resume (skills, projects, strengths)

Output format:
SUBJECT: <short subject line, <= 80 chars>
BODY:
<email body text, 6–10 sentences>
SIGNOFF: <candidate name placeholder {candidate_name}>

Tone:
- Professional, polite, confident
- Do NOT sound desperate
- Emphasize relevant skills/experience from resume
- Clearly state that the candidate is interested in opportunities matching the JD
- End with a polite request for further discussion
"""
                    # Try to heuristically get candidate name from resume (very basic)
                    import re
                    name_guess = ""
                    m = re.search(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})", resume_text.strip(), re.MULTILINE)
                    if m:
                        name_guess = m.group(1)

                    prompt = PromptTemplate(
                        input_variables=["job_snippets", "resume_snippets", "candidate_name"],
                        template=prompt_template
                    )

                    chain = LLMChain(llm=llm, prompt=prompt)

                    # Combine snippets for template
                    job_snips_text = "\n\n---\n\n".join(jd_snips) if jd_snips else job_text[:2000]
                    resume_snips_text = "\n\n---\n\n".join(resume_snips) if resume_snips else resume_text[:2000]

                    try:
                        result = chain.run({
                            "job_snippets": job_snips_text,
                            "resume_snippets": resume_snips_text,
                            "candidate_name": name_guess or "Candidate"
                        })
                    except Exception as e:
                        st.error(f"LLM call failed: {e}")
                        result = None

                    if result:
                        output_area.code(result)
                        # Provide download button
                        st.download_button("Download email (.txt)", data=result, file_name="cold_email.txt", mime="text/plain")
                        st.success("Cold email generated — tweak the prompt or retriever 'k' if you'd like different results.")
