from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os, shutil, uuid

from rag import build_rag_pipeline, query_rag

load_dotenv()

app = FastAPI(title="PDF RAG Chatbot", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory store: session_id -> rag chain
rag_sessions: dict = {}

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())
    save_path = f"uploads/{session_id}.pdf"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        rag_chain = build_rag_pipeline(save_path)
        rag_sessions[session_id] = rag_chain
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return {"session_id": session_id, "filename": file.filename, "status": "ready"}


class QueryRequest(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
async def chat(req: QueryRequest):
    if req.session_id not in rag_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = query_rag(rag_sessions[req.session_id], req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    rag_sessions.pop(session_id, None)
    pdf_path = f"uploads/{session_id}.pdf"
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {"status": "ok", "groq_key_set": bool(os.getenv("GROQ_API_KEY"))}
