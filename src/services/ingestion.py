import fitz  # PyMuPDF
from typing import List
from sqlalchemy.orm import Session
from src.database.models import DocumentChunk
from src.database.session import get_session
from src.services.embeddings import embed_texts


# -----------------------------
# Text Chunking Utility
# -----------------------------

def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200
) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# -----------------------------
# PDF Ingestion Pipeline
# -----------------------------

def ingest_pdf(
    pdf_path: str,
    document_id: str
) -> None:
    doc = fitz.open(pdf_path)
    session: Session = get_session()

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()

        if not text.strip():
            continue

        chunks = chunk_text(text)

        embeddings = embed_texts(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            record = DocumentChunk(
                content=chunk,
                embedding=embedding,
                document_id=document_id,
                page_number=page_number,
            )
            session.add(record)

    session.commit()
    session.close()
