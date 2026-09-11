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
from src.rag.vector_store import (
    add_chunks as vector_add_chunks,
    delete_doc as vector_delete_doc,
    list_all_documents,
)
from src.rag.keyword_index import (
    add_chunks as keyword_add_chunks,
    delete_doc as keyword_delete_doc,
)

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

# In-memory status tracker — ONLY for tracking in-flight background
# processing ("processing" / "error" states) within this process's
# lifetime. It is NOT the source of truth for what's indexed — that's
# Qdrant (see list_all_documents()). This dict is expected to be empty
# after every restart; that's fine, because /documents no longer reads
# from it for "ready" documents.
_upload_status: dict[str, dict] = {}
_cancelled_uploads: set[str] = set()


def _upload_was_cancelled(doc_id: str) -> bool:
    return doc_id in _cancelled_uploads


def _cleanup_cancelled_upload(doc_id: str, saved_path: Path) -> None:
    vector_delete_doc(doc_id)
    keyword_delete_doc(doc_id)
    if saved_path.exists():
        saved_path.unlink()


def _process_document(doc_id: str, saved_path: Path, ext: str, filename: str):
    try:
        if _upload_was_cancelled(doc_id):
            _cleanup_cancelled_upload(doc_id, saved_path)
            _cancelled_uploads.discard(doc_id)
            return

        parser = PARSERS[ext]
        with time_stage("parsing", {"ext": ext, "filename": filename}):
            parsed = parser(str(saved_path))

        if _upload_was_cancelled(doc_id):
            _cleanup_cancelled_upload(doc_id, saved_path)
            _cancelled_uploads.discard(doc_id)
            return

        chunk_texts = []
        if ext == ".xlsx":
            for sheet in parsed:
                sheet_chunks = chunk_text(sheet["text"])
                chunk_texts.extend(f"[Sheet: {sheet['sheet_name']}]\n{c.text}" for c in sheet_chunks)
        elif ext == ".pdf":
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
            if saved_path.exists():
                saved_path.unlink()
            return

        with time_stage("embedding_and_vector_index", {"n_chunks": len(chunk_texts)}):
            vector_count = vector_add_chunks(chunk_texts, doc_id=doc_id, source_filename=filename)
        if _upload_was_cancelled(doc_id):
            _cleanup_cancelled_upload(doc_id, saved_path)
            _cancelled_uploads.discard(doc_id)
            return

        with time_stage("keyword_index", {"n_chunks": len(chunk_texts)}):
            keyword_count = keyword_add_chunks(chunk_texts, doc_id=doc_id, source_filename=filename)

        if _upload_was_cancelled(doc_id):
            _cleanup_cancelled_upload(doc_id, saved_path)
            _cancelled_uploads.discard(doc_id)
            return

        _upload_status[doc_id] = {
            "status": "ready",
            "filename": filename,
            "chunks_indexed": vector_count,
            "keyword_indexed": keyword_count,
        }
    except Exception as e:
        if _upload_was_cancelled(doc_id):
            if saved_path.exists():
                saved_path.unlink()
            _cancelled_uploads.discard(doc_id)
            return
        logger.error(f"Background processing failed for {filename}: {e}")
        _upload_status[doc_id] = {"status": "error", "detail": str(e)}
        if saved_path.exists():
            saved_path.unlink()


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in PARSERS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    doc_id = str(uuid.uuid4())
    _cancelled_uploads.discard(doc_id)
    saved_path = UPLOAD_DIR / f"{doc_id}{ext}"
    contents = await file.read()
    saved_path.write_bytes(contents)

    _upload_status[doc_id] = {"status": "processing", "filename": file.filename}
    background_tasks.add_task(_process_document, doc_id, saved_path, ext, file.filename)

    return {"doc_id": doc_id, "filename": file.filename, "status": "processing"}


@router.get("/documents")
async def list_documents():
    """
    Qdrant is the source of truth for what's actually indexed and
    queryable — it survives restarts, _upload_status does not. Merge
    in only the in-flight "processing" entries from this process's
    memory (documents not yet in Qdrant), so an upload in progress
    still shows up before it's finished indexing.
    """
    persisted = {d["doc_id"]: d for d in list_all_documents()}

    docs = [
        {
            "doc_id": d["doc_id"],
            "filename": d["filename"],
            "chunks_indexed": d["chunk_count"],
        }
        for d in persisted.values()
    ]

    for doc_id, v in _upload_status.items():
        if doc_id in persisted:
            continue  # already indexed, already listed above
        if v.get("status") == "processing":
            docs.append({"doc_id": doc_id, "filename": v["filename"], "chunks_indexed": 0, "status": "processing"})

    return docs


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Deletes by doc_id directly against the persistent stores — does NOT
    require an in-memory _upload_status entry, so documents that survived
    a backend restart can still be deleted from the frontend.
    """
    _cancelled_uploads.add(doc_id)

    try:
        vector_deleted = vector_delete_doc(doc_id)
    except Exception as e:
        logger.error(f"Vector delete failed for {doc_id}: {e}")
        vector_deleted = False

    try:
        keyword_deleted = keyword_delete_doc(doc_id)
    except Exception as e:
        logger.error(f"Keyword delete failed for {doc_id}: {e}")
        keyword_deleted = False

    if not vector_deleted or not keyword_deleted:
        logger.error(
            f"Partial document deletion for {doc_id}: "
            f"vector_deleted={vector_deleted}, keyword_deleted={keyword_deleted}"
        )

    # Find the uploaded file on disk regardless of extension — its
    # filename isn't known from _upload_status after a restart.
    for candidate in UPLOAD_DIR.glob(f"{doc_id}.*"):
        candidate.unlink()

    _upload_status.pop(doc_id, None)

    if not vector_deleted and not keyword_deleted:
        raise HTTPException(status_code=404, detail="Unknown doc_id.")

    return {"doc_id": doc_id, "deleted": True}


@router.get("/upload/status/{doc_id}")
async def get_upload_status(doc_id: str):
    status = _upload_status.get(doc_id)
    if status is None:
        # Not in in-memory tracker — check if it's already fully indexed
        # (e.g. this process restarted after upload completed elsewhere).
        persisted = {d["doc_id"] for d in list_all_documents()}
        if doc_id in persisted:
            return {"doc_id": doc_id, "status": "ready"}
        raise HTTPException(status_code=404, detail="Unknown doc_id.")
    return {"doc_id": doc_id, **status}