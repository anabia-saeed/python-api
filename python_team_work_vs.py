from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pathlib import Path
from typing import List
import re
import requests
import uvicorn

# Optional document readers
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None
try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

# BASE_DIR = Path(__file__).resolve().parent
# UPLOAD_FOLDER = BASE_DIR / "uploads"
# UPLOAD_FOLDER.mkdir(exist_ok=True)
import os
from pathlib import Path

UPLOAD_FOLDER = Path(os.environ.get("UPLOAD_FOLDER", "/tmp/uploads"))
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = BASE_DIR
DATABASE_URL = f"sqlite:///{(BASE_DIR / 'documents.db').as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content = Column(Text, nullable=False)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DocSecure RAG Backend", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AskQuestion(BaseModel):
    document_id: int
    question: str

class AskAllQuestion(BaseModel):
    question: str
    document_ids: List[int] = []

SUPPORTED = {".txt", ".pdf", ".docx", ".xlsx", ".csv"}

def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".csv"}:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        if PdfReader is None:
            raise HTTPException(500, "PDF support is not installed. Run: python -m pip install pypdf")
        import io
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        if DocxDocument is None:
            raise HTTPException(500, "DOCX support is not installed. Run: python -m pip install python-docx")
        import io
        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext == ".xlsx":
        if load_workbook is None:
            raise HTTPException(500, "XLSX support is not installed. Run: python -m pip install openpyxl")
        import io
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        lines = []
        for ws in wb.worksheets:
            lines.append(f"Sheet: {ws.title}")
            for row in ws.iter_rows(values_only=True):
                values = [str(v) for v in row if v is not None]
                if values:
                    lines.append(" | ".join(values))
        return "\n".join(lines)
    raise HTTPException(400, f"Unsupported file type: {ext}. Use TXT, PDF, DOCX, XLSX or CSV.")

def chunks(text: str, size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    result = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end]
        if end < len(text):
            last = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "))
            if last > size * 0.55:
                end = start + last + 1
                chunk = text[start:end]
        result.append(chunk.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return result

def retrieve(question: str, documents, top_k=4):
    candidates = []
    for doc in documents:
        for chunk in chunks(doc.content):
            candidates.append((doc, chunk))
    if not candidates:
        return []
    if TfidfVectorizer is not None and cosine_similarity is not None:
        texts = [c[1] for c in candidates]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [(candidates[i][0], candidates[i][1], float(scores[i])) for i in order if scores[i] > 0]
    q_words = set(re.findall(r"\b\w+\b", question.lower()))
    scored = []
    for doc, chunk in candidates:
        words = set(re.findall(r"\b\w+\b", chunk.lower()))
        scored.append((doc, chunk, len(q_words & words)))
    return sorted(scored, key=lambda x: x[2], reverse=True)[:top_k]

def extractive_answer(question: str, contexts):
    if not contexts:
        return "I couldn't find relevant information in the uploaded documents. Try using terms that appear in the document."
    q_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", question.lower()))
    sentences = []
    for _, context, _ in contexts:
        for s in re.split(r"(?<=[.!?])\s+", context):
            if s.strip():
                words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", s.lower()))
                score = len(q_words & words)
                sentences.append((score, s.strip()))
    sentences.sort(key=lambda x: x[0], reverse=True)
    selected = [s for score, s in sentences if score > 0][:3]
    if not selected:
        selected = [contexts[0][1][:900]]
    return " ".join(selected)

def ollama_answer(question: str, contexts):
    """Optional local generation. If Ollama is running, use it; otherwise use extractive RAG."""
    if not contexts:
        return extractive_answer(question, contexts), "extractive"
    context = "\n\n".join(f"SOURCE: {doc.filename}\n{chunk}" for doc, chunk, _ in contexts)
    prompt = f"""Answer the question using ONLY the supplied sources. If the sources do not contain the answer, say you could not find it. Be concise and do not invent facts.\n\nSOURCES:\n{context}\n\nQUESTION: {question}\nANSWER:"""
    try:
        r = requests.post("http://127.0.0.1:11434/api/generate", json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}, timeout=12)
        if r.ok:
            return r.json().get("response", "").strip(), "ollama"
    except requests.RequestException:
        pass
    return extractive_answer(question, contexts), "extractive"

@app.get("/")
def home():
    return {"message": "DocSecure RAG backend is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "rag": "tfidf", "generator": "ollama-optional"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Please select a file.")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, "Supported files: TXT, PDF, DOCX, XLSX, CSV")
    data = await file.read()
    if not data:
        raise HTTPException(400, "File is empty.")
    text = extract_text(file.filename, data)
    if not text.strip():
        raise HTTPException(400, "No readable text was found in this document.")
    safe_name = Path(file.filename).name
    (UPLOAD_FOLDER / safe_name).write_bytes(data)
    db = SessionLocal()
    try:
        doc = Document(filename=safe_name, content=text)
        db.add(doc); db.commit(); db.refresh(doc)
        return {"message": "Document uploaded and indexed", "document_id": doc.id, "filename": doc.filename, "characters": len(text)}
    finally:
        db.close()

@app.get("/documents")
def get_documents():
    db = SessionLocal()
    try:
        docs = db.query(Document).order_by(Document.id.desc()).all()
        return [{"id": d.id, "filename": d.filename, "characters": len(d.content)} for d in docs]
    finally:
        db.close()

@app.get("/documents/{document_id}")
def get_document(document_id: int):
    db = SessionLocal()
    try:
        d = db.query(Document).filter(Document.id == document_id).first()
        if not d: raise HTTPException(404, "Document not found")
        return {"id": d.id, "filename": d.filename, "content": d.content}
    finally:
        db.close()

@app.get("/search")
def search_documents(keyword: str):
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        hits = []
        for d in docs:
            if keyword.lower() in d.content.lower() or keyword.lower() in d.filename.lower():
                hits.append({"id": d.id, "filename": d.filename})
        return hits
    finally:
        db.close()

@app.get("/count")
def count_documents():
    db = SessionLocal()
    try: return {"total_documents": db.query(Document).count()}
    finally: db.close()

@app.post("/ask-ai")
def ask_ai(request: AskQuestion):
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == request.document_id).first()
        if not doc: raise HTTPException(404, "Document not found")
        contexts = retrieve(request.question, [doc])
        answer, generator = ollama_answer(request.question, contexts)
        return {"document": doc.filename, "question": request.question, "answer": answer, "generator": generator,
                "sources": [{"filename": d.filename, "score": round(score, 3), "snippet": chunk[:300]} for d, chunk, score in contexts]}
    finally: db.close()

@app.post("/ask-all")
def ask_all(request: AskAllQuestion):
    db = SessionLocal()
    try:
        docs = db.query(Document).all() if not request.document_ids else db.query(Document).filter(Document.id.in_(request.document_ids)).all()
        contexts = retrieve(request.question, docs)
        answer, generator = ollama_answer(request.question, contexts)
        return {"question": request.question, "answer": answer, "generator": generator,
                "sources": [{"filename": d.filename, "score": round(score, 3), "snippet": chunk[:300]} for d, chunk, score in contexts]}
    finally: db.close()

@app.delete("/documents/{document_id}")
def delete_document(document_id: int):
    db = SessionLocal()
    try:
        d = db.query(Document).filter(Document.id == document_id).first()
        if not d: raise HTTPException(404, "Document not found")
        db.delete(d); db.commit()
        path = UPLOAD_FOLDER / d.filename
        if path.exists(): path.unlink()
        return {"message": "Document deleted successfully"}
    finally: db.close()

# Serve the HTML frontend directly from FastAPI. Open http://127.0.0.1:8000/ in the browser.
@app.get("/{page:path}")
def frontend(page: str):
    if page in {"", "/"}: page = "index.html"
    target = (FRONTEND_DIR / page).resolve()
    if target.is_file() and str(target).startswith(str(FRONTEND_DIR.resolve())):
        return FileResponse(target)
    raise HTTPException(404, "Page not found")

if __name__ == "__main__":
    print("DocSecure RAG backend: http://127.0.0.1:8000")
    print("Swagger API: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
