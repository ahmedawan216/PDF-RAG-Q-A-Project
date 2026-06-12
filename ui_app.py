import os
import re
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

def sanitize_collection_name(name:str) -> str:
   name = name.lower()
   name = name.replace(".pdf", "")
   name = re.sub(r"[^a-z0-9._-]", "_", name)
   name = re.sub(r"_+", "_", name)
   return name.strip("_")[:50]


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

def render_message(role, content):
    if role == "user":
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; margin:8px 0;">
            <div style="background:#e8f4f8; border-radius:12px; 
                        padding:10px 14px; max-width:75%;">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-start; margin:8px 0;">
            <div style="background:#f0f0f0; border-radius:12px; 
                        padding:10px 14px; max-width:75%;">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)

def build_rag_pipeline(docs, collection_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    summary_chunk = Document(
       page_content=f"Document summary from first page:\n{docs[0].page_content}",
       metadata={"source": "page_1_summary", "page": 0}
    )
    chunks.insert(0, summary_chunk)

    print(f"Total chunks created: {len(chunks)}")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 8}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    )

    system_prompt = """You are a document analyst assistant. Your job is to answer questions strictly based on the document context provided below.

Rules:
- Only use information present in the context. Never use outside knowledge.
- If the answer is clearly in the context, answer directly and concisely.
- If the context partially answers the question, share what you found and note what's missing.
- If the answer is not in the context at all, say: "I couldn't find this in the provided document."
- Do not make up names, dates, or facts.

Context:
{context}
"""

    user_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(
        llm,
        user_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        doc_chain
    )

    return retrieval_chain


st.set_page_config(
    page_title="PDF RAG",
    page_icon="📄"
)

heading = st.sidebar.header("Document Control")
uploaded_file = st.sidebar.file_uploader("Upload your target document", type=["pdf"])

if "messages" not in st.session_state:
   st.session_state.messages = []

st.title("📄 Chat with your PDF Assistant")
st.text("Ask anything about the document you uploaded in the sidebar panel.")

if uploaded_file is not None:
  st.sidebar.success("PDF Uploaded Successfully!")
  
  if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
    st.session_state.current_file = uploaded_file.name

  with st.spinner("Processing PDF"):
      with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

        pipeline_loader = PyPDFLoader(temp_path)
        docs = pipeline_loader.load()

        collection_name = sanitize_collection_name(
           uploaded_file.name
        )

        retrieval_chain = build_rag_pipeline(
           docs,
           collection_name
        )

        st.sidebar.success("RAG Pipeline Ready for Queries!")
        st.sidebar.info(f"Loaded {len(docs)} pages from the document context.")

      os.unlink(temp_path)

  # Render existing chat history
  for message in st.session_state.messages:
      render_message(message["role"], message["content"])

  if user_question := st.chat_input("Ask a question about your PDF..."):
        render_message("user", user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.spinner("Analyzing document context..."):
          results = retrieval_chain.invoke({"input": user_question})
          answer = results["answer"]

        render_message("assistant", answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        