# AI-Cold-Mail-Bot
An AI-powered bot that helps job seekers create **professional job application emails**.   Upload your **Resume (PDF)** and a **Job Description (URL or PDF)**, and the bot will generate a tailored cold mail highlighting your skills that match the role.

---

## 🚀 Features
- 🔹 Extracts important details from **Job Description** (skills, role, company).
- 🔹 Reads **Resume** and highlights relevant skills/projects.
- 🔹 Generates a **professional cold mail** using Generative AI.
- 🔹 Built with **Streamlit**, **LangChain**, and **Groq LLM**.
- 🔹 Supports both **URL-based JD** and **PDF upload**.

---


## 🤖 Bot Workflow

1. **Input Job Description**  
   - User provides a **Job Description** either via **URL** or **PDF upload** (only one).  

2. **Upload Resume**  
   - User uploads their **Resume (PDF)**.  

3. **Text Extraction**  
   - Bot extracts text from the **Job Description** and **Resume**.  

4. **Vectorization (Optional)**  
   - Text chunks are converted into **vector embeddings** for better context retrieval.  

5. **Candidate & Job Info Extraction**  
   - Bot identifies key details such as:
     - Company name  
     - Job title  
     - Required skills  
     - Candidate name  
     - Candidate skills and projects  

6. **Generate Cold Mail**  
   - Using **Groq LLM**, the bot creates a **personalized, professional cold email**:
     - Includes company name, role, and matching skills
     - Highlights candidate projects or experience
     - Ends with a polite call to action  

7. **Output**  
   - Generated email is displayed on the UI

---


<img width="1919" height="833" alt="Screenshot 2025-08-27 140656" src="https://github.com/user-attachments/assets/c9656481-9f25-4b44-911e-7e39dabce358" />

---
<img width="1919" height="859" alt="Screenshot 2025-08-27 140751" src="https://github.com/user-attachments/assets/05d8a5c9-f940-4e38-8542-b80475a22cc7" />

---
<img width="1919" height="851" alt="Screenshot 2025-08-27 140805" src="https://github.com/user-attachments/assets/a5f85388-fa6a-4273-95e4-26e7b4051699" />

---
<img width="1919" height="829" alt="Screenshot 2025-08-27 140816" src="https://github.com/user-attachments/assets/8d81740c-12f7-4559-a068-65a2daaaba7e" />

---

## 🛠️ Tech Stack
- **Python 3.9+**
- [Streamlit](https://streamlit.io/) – for the web interface
- [LangChain](https://www.langchain.com/) – for document parsing & chaining
- [FAISS](https://github.com/facebookresearch/faiss) – vector storage (optional for similarity search)
- [Groq LLM API](https://groq.com/) – LLM for mail generation
- **PDF Processing** – `PyPDFLoader`, `UnstructuredURLLoader`

---

## 📂 Project Structure
