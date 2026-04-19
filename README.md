# 📄 PDF RAG Chatbot

Ask natural-language questions about any PDF — powered by **FastAPI**, **LangChain**, **FAISS**, and **Groq API (LLaMA 3)**.

Upload a PDF → it gets chunked & embedded into a FAISS vector store → you ask a question → relevant chunks are retrieved → Groq's LLaMA 3 generates the answer. Full RAG pipeline.

---

## 🗂️ Project Structure

```
pdf_rag_chatbot/
├── main.py              ← FastAPI app (routes: /, /upload, /chat, /health)
├── rag.py               ← RAG pipeline (PDF load → chunk → embed → FAISS → QA)
├── templates/
│   └── index.html       ← Full HTML/CSS/JS frontend (no framework needed)
├── static/              ← (empty — place CSS/JS assets here if needed)
├── uploads/             ← PDFs saved here temporarily per session
├── requirements.txt     ← All Python dependencies
├── .env.example         ← API key template
├── .gitignore
└── README.md
```

---

## 🚀 Run in VS Code — Step by Step

### 1. Open the project in VS Code
```bash
code pdf_rag_chatbot
```
Or: **File → Open Folder** → select the `pdf_rag_chatbot` folder.

---

### 2. Open the integrated terminal
**Keyboard shortcut:** `Ctrl + `` ` (backtick)  
Or: **Terminal → New Terminal** from the menu bar.

---

### 3. Create a virtual environment
```bash
python -m venv venv
```

---

### 4. Activate the virtual environment

**Windows (Command Prompt / PowerShell):**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

> ✅ You should see `(venv)` at the start of your terminal prompt.

---

### 5. Install all dependencies
```bash
pip install -r requirements.txt
```
> ⏳ First install takes 2–4 minutes (downloads sentence-transformers model).

---

### 6. Set up your Groq API key

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` in VS Code and replace the placeholder:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
Get your **free** key at 👉 **https://console.groq.com**

---

### 7. Run the server
```bash
uvicorn main:app --reload
```

Open your browser at 👉 **http://localhost:8000**

---

## ✅ How to Use

1. Click **"Click to upload"** or drag & drop a PDF
2. Wait for "✅ PDF ready!" confirmation
3. Type any question in the chat input
4. Get answers sourced directly from your document!

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Serve the frontend UI |
| `POST` | `/upload` | Upload a PDF → returns `session_id` |
| `POST` | `/chat`  | Send question → get RAG answer |
| `DELETE` | `/session/{id}` | Clear a session |
| `GET`  | `/health` | Check API + Groq key status |
| `GET`  | `/docs`  | Auto-generated Swagger UI |

---

## ⚙️ How the RAG Pipeline Works

```
PDF file
  ↓  PyPDFLoader
Raw text pages
  ↓  RecursiveCharacterTextSplitter (chunk_size=800, overlap=100)
Text chunks
  ↓  HuggingFace sentence-transformers (all-MiniLM-L6-v2) — runs locally, free
Vector embeddings
  ↓  FAISS vector store (in-memory, per session)
Top-4 relevant chunks (cosine similarity)
  ↓  Groq API — LLaMA 3 8B
Final answer
```

---

## 🚢 Deploy Free (Render.com)

1. Push to GitHub
2. Go to **https://render.com** → New Web Service
3. Connect your repo, set **Start Command:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
4. Add environment variable: `GROQ_API_KEY = your_key`
5. Deploy!

---

Built by **Shruti Umakant Rede** , Pune  
Stack: FastAPI · LangChain · FAISS · Groq API · HuggingFace · Python
