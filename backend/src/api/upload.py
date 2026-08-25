"""
POST /upload — accepts a file, parses it, chunks it, indexes it into
both the vector store and keyword index under a generated doc_id.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from src.utils.timing import time_stage
from src.ingestion.parsers.pdf_parser import parse_pdf
from src.ingestion.parsers.docx_parser import parse_docx
from src.ingestion.parsers.excel_parser import parse_excel
from src.ingestion.parsers.csv_parser import parse_csv
from src.ingestion.parsers.image_parser import parse_image
from src.ingestion.chunker import chunk_text
from src.rag.vector_store import add_chunks as vector_add_chunks
from src.rag.keyword_index import add_chunks as keyword_add_chunks

logger = logging.getLogger(__name__)
router = APIRouter()

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_excel,
    ".csv": parse_csv,
    ".jpg": parse_image,
    ".jpeg": parse_image,
    ".png": parse_image,
}

UPLOAD_DIR = Path("./uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory status tracker
_upload_status: dict[str, dict] = {}


def _process_document(doc_id: str, saved_path: Path, ext: str, filename: str):
    try:
        parser = PARSERS[ext]
        with time_stage("parsing", {"ext": ext, "filename": filename}):
            parsed = parser(str(saved_path))

        chunk_texts = []
        if ext == ".xlsx":
            for sheet in parsed:
                sheet_chunks = chunk_text(sheet["text"])
                chunk_texts.extend(f"[Sheet: {sheet['sheet_name']}]\n{c.text}" for c in sheet_chunks)
        elif ext == ".pdf":
            # Parsed PDFs return a dict with 'sections' (list of {section_title, page_num, text})
            if isinstance(parsed, dict) and "sections" in parsed:
                for sec in parsed["sections"]:
                    sec_chunks = chunk_text(sec["text"])
                    for c in sec_chunks:
                        title = sec.get("section_title") or ""
                        page = sec.get("page_num")
                        prefix = f"[Section: {title}] [Page: {page}]\n" if title or page else ""
                        chunk_texts.append(prefix + c.text)
            else:
                chunks = chunk_text(parsed if isinstance(parsed, str) else parsed.get("full_text", ""))
                chunk_texts = [c.text for c in chunks]
        else:
            chunks = chunk_text(parsed)
            chunk_texts = [c.text for c in chunks]

        if not chunk_texts:
            _upload_status[doc_id] = {"status": "error", "detail": "No extractable text found in the file."}
            return

        with time_stage("embedding_and_vector_index", {"n_chunks": len(chunk_texts)}):
            vector_count = vector_add_chunks(chunk_texts, doc_id=doc_id, source_filename=filename)
        with time_stage("keyword_index", {"n_chunks": len(chunk_texts)}):
            keyword_count = keyword_add_chunks(chunk_texts, doc_id=doc_id, source_filename=filename)

        _upload_status[doc_id] = {
            "status": "ready",
            "filename": filename,
            "chunks_indexed": vector_count,
            "keyword_indexed": keyword_count,
        }
    except Exception as e:
        logger.error(f"Background processing failed for {filename}: {e}")
        _upload_status[doc_id] = {"status": "error", "detail": str(e)}


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in PARSERS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    doc_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{doc_id}{ext}"
    contents = await file.read()
    saved_path.write_bytes(contents)

    _upload_status[doc_id] = {"status": "processing", "filename": file.filename}
    background_tasks.add_task(_process_document, doc_id, saved_path, ext, file.filename)

    return {"doc_id": doc_id, "filename": file.filename, "status": "processing"}


@router.get("/documents")
async def list_documents():
    docs = [
        {"doc_id": doc_id, "filename": v["filename"], "chunks_indexed": v.get("chunks_indexed", 0)}
        for doc_id, v in _upload_status.items()
        if v.get("status") == "ready"
    ]
    return docs


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    from src.rag.vector_store import delete_doc as vector_delete_doc
    from src.rag.keyword_index import delete_doc as keyword_delete_doc
    if doc_id not in _upload_status:
        raise HTTPException(status_code=404, detail="Unknown doc_id.")
    vector_delete_doc(doc_id)
    keyword_delete_doc(doc_id)
    del _upload_status[doc_id]
    return {"doc_id": doc_id, "deleted": True}


@router.get("/upload/status/{doc_id}")
async def get_upload_status(doc_id: str):
    status = _upload_status.get(doc_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown doc_id.")
    return {"doc_id": doc_id, **status}