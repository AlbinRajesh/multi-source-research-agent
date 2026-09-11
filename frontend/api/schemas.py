from pydantic import BaseModel
from typing import Optional


class ResearchRequest(BaseModel):
    query: str
    doc_ids: Optional[list[str]] = None  # explicit doc selection — None/empty = web-only


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    uploaded_at: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool