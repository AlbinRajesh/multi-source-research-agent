import { useRef, useEffect, useState } from "react";
import {
  uploadFile,
  deleteDocument,
  getUploadStatus,
  listDocuments,
} from "../api/upload";

const EXT_STYLES = {
  pdf: { icon: "picture_as_pdf", bg: "bg-error-container/20", fg: "text-error" },
  docx: { icon: "description", bg: "bg-primary-fixed", fg: "text-primary" },
  xlsx: { icon: "table_chart", bg: "bg-tertiary-fixed", fg: "text-tertiary" },
  csv: { icon: "table_chart", bg: "bg-tertiary-fixed", fg: "text-tertiary" },
};

export default function FileUploader({ docs, selectedDocId, onSelectDoc, setDocs }) {
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    listDocuments()
      .then(setDocs)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load documents."));
  }, [setDocs]);

  const inputRef = useRef(null);

  async function handleUpload(file) {
    setError(null);
    setUploading(true);
    let uploadedDocId = null;
    try {
      const res = await uploadFile(file);
      uploadedDocId = res.doc_id;
      setDocs((prev) => [
        ...prev,
        {
          doc_id: res.doc_id,
          filename: res.filename,
          chunks_indexed: 0,
          status: "processing",
        },
      ]);

      let completed = false;
      for (let attempt = 0; attempt < 300; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const status = await getUploadStatus(res.doc_id);
        if (status.status === "ready") {
          setDocs((prev) =>
            prev.map((doc) =>
              doc.doc_id === res.doc_id
                ? {
                    ...doc,
                    filename: status.filename,
                    chunks_indexed: status.chunks_indexed,
                    status: "ready",
                  }
                : doc
            )
          );
          completed = true;
          break;
        }
        if (status.status === "error") {
          throw new Error(status.detail || "Document processing failed.");
        }
      }
      if (!completed) {
        throw new Error("Document processing timed out.");
      }
    } catch (err) {
      if (uploadedDocId) {
        setDocs((prev) => prev.filter((doc) => doc.doc_id !== uploadedDocId));
      }
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  function handleFileSelect(e) {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleUpload(file);
  }

  async function handleDelete(docId) {
    setError(null);
    try {
      await deleteDocument(docId);
      setDocs((prev) => prev.filter((d) => d.doc_id !== docId));
      if (docId === selectedDocId) {
        onSelectDoc(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed.");
    }
  }

  return (
    <div className="flex flex-col gap-xl w-full">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-headline-sm text-on-surface">Your Documents</h2>
          <p className="font-label-sm text-on-surface-variant">
            Click a document to scope chat to only that file.
          </p>
        </div>
      </div>

      <label
        className={`group relative p-lg rounded-xl bg-surface-container-lowest border-2 border-dashed transition-all cursor-pointer flex flex-col items-center gap-md text-center w-full ${
          dragging ? "border-primary" : "border-outline-variant hover:border-primary/50"
        } ${uploading ? "opacity-60 pointer-events-none" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.xlsx,.csv,.jpg,.jpeg,.png"
          onChange={handleFileSelect}
          disabled={uploading}
          className="hidden"
        />
        <div className="w-12 h-12 rounded-full bg-primary/5 flex items-center justify-center text-primary group-hover:scale-110 transition-transform">
          <span className="material-symbols-outlined text-[28px]">
            {uploading ? "hourglass_top" : "cloud_upload"}
          </span>
        </div>
        <div>
          <p className="font-label-md text-on-surface">
            {uploading ? "Indexing…" : "Click or drag to upload"}
          </p>
          <p className="font-label-sm text-on-surface-variant">
            PDF, DOCX, XLSX, CSV
          </p>
        </div>
      </label>

      {error && (
        <div className="px-md py-sm rounded-lg bg-error-container text-on-error-container font-label-sm break-words">
          {error}
        </div>
      )}

      <div className="flex flex-col gap-md w-full">
        <div className="flex items-center justify-between px-xs">
          <span className="font-label-sm uppercase tracking-widest text-on-surface-variant/70">
            Indexed Files
          </span>
        </div>

        {docs.length === 0 ? (
          <p className="font-label-sm text-on-surface-variant px-xs">
            No documents indexed yet.
          </p>
        ) : (
          <div className="flex flex-col gap-sm w-full">
            {docs.map((d) => {
              const ext = d.filename.split(".").pop()?.toLowerCase() ?? "";
              const style = EXT_STYLES[ext] ?? {
                icon: "draft",
                bg: "bg-surface-container-highest",
                fg: "text-on-surface-variant",
              };
              const isSelected = d.doc_id === selectedDocId;
              return (
                <div
                  key={d.doc_id}
                  onClick={() =>
                    onSelectDoc(isSelected ? null : d.doc_id)
                  }
                  className={`group flex items-center gap-sm p-md rounded-lg transition-all shadow-sm cursor-pointer w-full min-w-0 ${
                    isSelected
                      ? "border-2 border-primary bg-primary/10"
                      : "bg-surface-container-lowest hover:bg-surface-container-high"
                  }`}
                  aria-selected={isSelected}
                >
                  <div
                    className={`w-10 h-10 shrink-0 rounded flex items-center justify-center ${style.bg} ${style.fg}`}
                  >
                    <span className="material-symbols-outlined text-[20px]">
                      {style.icon}
                    </span>
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 min-w-0">
                      <p className="font-label-md text-on-surface truncate min-w-0">
                        {d.filename}
                      </p>
                      {isSelected && (
                        <span className="shrink-0 rounded-full border border-primary px-2 py-0.5 text-[11px] font-semibold text-primary">
                          Selected
                        </span>
                      )}
                    </div>
                    <p className="font-label-sm text-on-surface-variant">
                      {d.status === "processing"
                        ? "Indexing..."
                        : `${d.chunks_indexed} chunks`}
                    </p>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(d.doc_id);
                    }}
                    aria-label={`Remove ${d.filename}`}
                    title="Remove"
                    className="w-8 h-8 shrink-0 rounded-full flex items-center justify-center text-on-surface-variant opacity-100 sm:opacity-0 sm:group-hover:opacity-100 hover:bg-error-container hover:text-error transition-opacity"
                  >
                    <span className="material-symbols-outlined text-[18px]">close</span>
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}