# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

loader = PyPDFLoader('Profile (8).pdf')
docs = loader.load()

print(len(docs))
print(docs[0].page_content[:500])

splitter = RecursiveCharacterTextSplitter(
  chunk_size = 500,
  chunk_overlap = 100
)

chunks = splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
  model_name = "all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
  documents=chunks,
  embedding=embeddings,
  persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(
  search_kwargs = {"k":3}
)

llm = ChatGroq(
  model="llama-3.3-70b-versatile"
)

system_prompt = """ 
  You are an expert document assistant. Answer the user's question using only the provided context below.
    If you do not know the answer based on the context, say that you don't know.\n\n
    Context:\n{context}
  """

user_prompt = ChatPromptTemplate.from_messages([
  ("system", system_prompt),
  ("human", "{input}")
])

doc_chain = create_stuff_documents_chain(
  llm,
  user_prompt
)

retrieval_chain = create_retrieval_chain(retriever, doc_chain)

while True:
  user_question = input("\Ask something about your PDF (or type 'exit' to quit)")

  if user_question.lower() == 'exit':
    break
   
  # Run the pipeline once and catch the output dictionary
  results = retrieval_chain.invoke({"input": user_question})

  # Extract and print just the generated answer string field
  print("\n--- RAG Generated Answer ---")
  print(results["answer"])
