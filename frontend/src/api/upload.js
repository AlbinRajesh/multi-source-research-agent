const API_BASE = "http://localhost:8001";

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed with status ${res.status}`);
  }

  return res.json();
}

export async function listDocuments() {
  const res = await fetch(`${API_BASE}/documents`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to load documents (status ${res.status})`);
  }

  return res.json();
}

export async function deleteDocument(docId) {
  const res = await fetch(`${API_BASE}/documents/${docId}`, {
    method: "DELETE",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Delete failed with status ${res.status}`);
  }

  return res.json();
}

export async function getUploadStatus(docId) {
  const res = await fetch(`${API_BASE}/upload/status/${docId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Status check failed (${res.status})`);
  }
  return res.json();
}