from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List
import re
import requests

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

app = FastAPI(title="DocSecure RAG Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Aap yahan apna Netlify frontend URL bhi de sakte hain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document storage for Serverless environment (Vercel)
# Note: Serverless functions restart ho sakti hain, isliye production ke liye external DB (jaise Supabase) behtar hota hai.
fake_db_documents = []
document_counter = 1

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
            raise HTTPException(500, "PDF support is not installed.")
        import io
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        if DocxDocument is None:
            raise HTTPException(500, "DOCX support is not installed.")
        import io
        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext == ".xlsx":
        if load_workbook is None:
            raise HTTPException(500, "XLSX support is not installed.")
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
        for chunk in chunks(doc["content"]):
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
        return "I couldn't find relevant information in the uploaded documents."
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
    if not contexts:
        return extractive_answer(question, contexts), "extractive"
    context = "\n\n".join(f"SOURCE: {doc['filename']}\n{chunk}" for doc, chunk, _ in contexts)
    prompt = f"""Answer the question using ONLY the supplied sources. If the sources do not contain the answer, say you could not find it. Be concise and do not invent facts.\n\nSOURCES:\n{context}\n\nQUESTION: {question}\nANSWER:"""
    try:
        r = requests.post("http://127.0.0.1:11434/api/generate", json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}, timeout=12)
        if r.ok:
            return r.json().get("response", "").strip(), "ollama"
    except requests.RequestException:
        pass
    return extractive_answer(question, contexts), "extractive"

@app.get("/api")
def home():
    return {"message": "DocSecure RAG backend is running on Vercel"}

@app.get("/api/health")
def health():
    return {"status": "ok", "rag": "tfidf", "generator": "ollama-optional"}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global document_counter
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
    doc_id = document_counter
    document_counter += 1
    
    new_doc = {"id": doc_id, "filename": safe_name, "content": text}
    fake_db_documents.append(new_doc)
    
    return {"message": "Document uploaded and indexed", "document_id": doc_id, "filename": safe_name, "characters": len(text)}

@app.get("/api/documents")
def get_documents():
    return [{"id": d["id"], "filename": d["filename"], "characters": len(d["content"])} for d in reversed(fake_db_documents)]

@app.get("/api/documents/{document_id}")
def get_document(document_id: int):
    d = next((doc for doc in fake_db_documents if doc["id"] == document_id), None)
    if not d:
        raise HTTPException(404, "Document not found")
    return {"id": d["id"], "filename": d["filename"], "content": d["content"]}

@app.get("/api/search")
def search_documents(keyword: str):
    hits = []
    for d in fake_db_documents:
        if keyword.lower() in d["content"].lower() or keyword.lower() in d["filename"].lower():
            hits.append({"id": d["id"], "filename": d["filename"]})
    return hits

@app.get("/api/count")
def count_documents():
    return {"total_documents": len(fake_db_documents)}

@app.post("/api/ask-ai")
def ask_ai(request: AskQuestion):
    doc = next((d for d in fake_db_documents if d["id"] == request.document_id), None)
    if not doc:
        raise HTTPException(404, "Document not found")
    contexts = retrieve(request.question, [doc])
    answer, generator = ollama_answer(request.question, contexts)
    return {
        "document": doc["filename"], 
        "question": request.question, 
        "answer": answer, 
        "generator": generator,
        "sources": [{"filename": d["filename"], "score": round(score, 3), "snippet": chunk[:300]} for d, chunk, score in contexts]
    }

@app.post("/api/ask-all")
def ask_all(request: AskAllQuestion):
    docs = fake_db_documents if not request.document_ids else [d for d in fake_db_documents if d["id"] in request.document_ids]
    contexts = retrieve(request.question, docs)
    answer, generator = ollama_answer(request.question, contexts)
    return {
        "question": request.question, 
        "answer": answer, 
        "generator": generator,
        "sources": [{"filename": d["filename"], "score": round(score, 3), "snippet": chunk[:300]} for d, chunk, score in contexts]
    }

@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int):
    global fake_db_documents
    doc = next((d for d in fake_db_documents if d["id"] == document_id), None)
    if not doc:
        raise HTTPException(404, "Document not found")
    fake_db_documents = [d for d in fake_db_documents if d["id"] != document_id]
    return {"message": "Document deleted successfully"}
