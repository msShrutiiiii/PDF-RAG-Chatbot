import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ── Embedding model (free, runs locally) ───────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── LLM via Groq ───────────────────────────────────────────────────────────────
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env file.")
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=1024,
    )


def build_rag_pipeline(pdf_path: str):
    """Load PDF → chunk → embed → store in FAISS → return RetrievalQA chain."""

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError("Could not extract text from the PDF. It may be scanned/image-based.")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed chunks
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    # 4. Store in FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Build RetrievalQA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    return qa_chain


def query_rag(qa_chain, question: str) -> str:
    """Run a question through the RAG chain and return the answer."""
    result = qa_chain.invoke({"query": question})
    return result.get("result", "I could not find an answer in the document.")
