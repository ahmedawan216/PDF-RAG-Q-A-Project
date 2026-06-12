# 📄 PDF RAG Q&A Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions about their content.

The application extracts text from uploaded PDFs, converts the content into vector embeddings, stores them in a Chroma vector database, retrieves the most relevant document chunks, and uses a Large Language Model (LLM) to generate accurate answers based on the uploaded document.

---

## 🚀 Features

- Upload PDF documents through a Streamlit interface
- Automatic PDF text extraction using PyPDF
- Intelligent document chunking for efficient retrieval
- Semantic search using Hugging Face embeddings
- Vector storage with ChromaDB
- Context-aware question answering using Groq LLM
- Chat-style user interface
- Answers restricted to document content to reduce hallucinations

---

## 🏗️ Architecture

### Document Processing

1. Upload PDF document
2. Extract text using `PyPDFLoader`
3. Split document into chunks using `RecursiveCharacterTextSplitter`
4. Generate embeddings using Hugging Face Sentence Transformers
5. Store embeddings in ChromaDB

### Question Answering

1. User submits a question
2. Retriever finds the most relevant chunks
3. Retrieved context is sent to the LLM
4. LLM generates an answer based only on the retrieved content
5. Response is displayed in the chat interface

---

## 🛠️ Technologies Used

### Frontend
- Streamlit

### LLM
- Groq
- Llama 3.3 70B Versatile

### Vector Database
- ChromaDB

### Embeddings
- Hugging Face Embeddings
- all-MiniLM-L6-v2

### Frameworks
- LangChain
- LangChain Community
- LangChain Chroma
- LangChain Groq

### Document Processing
- PyPDF

---

## 📂 Project Structure

```text
PDF-RAG-Q-A-Project/
│
├── ui_app.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/ahmedawan216/PDF-RAG-Q-A-Project.git
cd PDF-RAG-Q-A-Project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Run Application

```bash
streamlit run ui_app.py
```

---

## 📸 Example Workflow

1. Upload a PDF document
2. Wait for document processing
3. Ask questions such as:

```text
What is the candidate's name?
What skills are mentioned?
Summarize the document.
What projects are listed?
```

4. Receive answers generated from the uploaded document context

---

## 🎯 Learning Objectives

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Semantic Search
- Document Question Answering
- LangChain Pipelines
- Prompt Engineering
- Streamlit Application Development

---

## 🔮 Future Improvements

- Support multiple PDFs
- Persistent vector storage
- Conversation memory
- Source citations
- Hybrid search
- Metadata filtering
- PDF preview functionality

---

## 👨‍💻 Author

Ahmed Awan

GitHub: https://github.com/ahmedawan216

---
