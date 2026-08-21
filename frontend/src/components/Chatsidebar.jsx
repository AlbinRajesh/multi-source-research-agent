import { useEffect, useRef, useState } from "react";
import { IconPlus, IconChat, IconDot, IconMore, IconEdit, IconTrash, IconCheck } from "./icons";

const STATUS_DOT = {
  running: "text-single animate-pulse",
  done: "text-verified",
  error: "text-conflict",
};

function SessionMenu({ onRename, onDelete, onClose }) {
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) onClose();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  return (
    <div
      ref={ref}
      className="absolute right-0 top-8 z-20 w-36 rounded-lg border border-border bg-surface-2 shadow-xl shadow-black/40 py-1 overflow-hidden"
    >
      <button
        onClick={onRename}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-ink-muted hover:bg-surface-3 hover:text-ink transition-colors"
      >
        <IconEdit /> Rename
      </button>
      <button
        onClick={onDelete}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-conflict hover:bg-surface-3 transition-colors"
      >
        <IconTrash /> Delete
      </button>
    </div>
  );
}

function SessionCard({ session, active, onSelect, onRename, onDelete }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.topic);
  const inputRef = useRef(null);

  useEffect(() => {
    if (editing) inputRef.current?.focus();
  }, [editing]);

  const commitRename = () => {
    const trimmed = draft.trim();
    if (trimmed) onRename(session.id, trimmed);
    setEditing(false);
  };

  return (
    <li className="relative group">
      <button
        onClick={() => !editing && onSelect(session.id)}
        className={`w-full flex items-start gap-2.5 rounded-lg px-3 py-2.5 text-left transition-all duration-200 ease-in-out border-l-2 ${
          active
            ? "bg-accent-soft/60 border-accent text-ink"
            : "border-transparent text-ink-muted hover:bg-surface-2 hover:text-ink"
        }`}
      >
        <IconChat className={`mt-0.5 shrink-0 ${active ? "text-accent-light" : ""}`} />

        <span className="flex-1 min-w-0">
          {editing ? (
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitRename();
                if (e.key === "Escape") setEditing(false);
              }}
              className="w-full bg-surface-3 rounded px-1.5 py-0.5 text-sm text-ink outline-none ring-1 ring-accent/40"
            />
          ) : (
            <span className="block text-sm truncate leading-snug">{session.topic}</span>
          )}
        </span>

        {!editing && (
          <IconDot className={`mt-1.5 shrink-0 ${STATUS_DOT[session.status] ?? "text-ink-faint"}`} />
        )}
      </button>

      {!editing && (
        <div
          className={`absolute right-1.5 top-1.5 transition-opacity duration-150 ${
            menuOpen ? "opacity-100" : "opacity-0 group-hover:opacity-100"
          }`}
        >
          <button
            onClick={(e) => {
              e.stopPropagation();
              setMenuOpen((v) => !v);
            }}
            className="w-6 h-6 flex items-center justify-center rounded-md text-ink-faint hover:bg-surface-3 hover:text-ink transition-colors"
          >
            <IconMore />
          </button>
          {menuOpen && (
            <SessionMenu
              onClose={() => setMenuOpen(false)}
              onRename={() => {
                setMenuOpen(false);
                setEditing(true);
              }}
              onDelete={() => {
                setMenuOpen(false);
                onDelete(session.id);
              }}
            />
          )}
        </div>
      )}

      {editing && (
        <button
          onClick={commitRename}
          className="absolute right-1.5 top-1.5 w-6 h-6 flex items-center justify-center rounded-md text-verified hover:bg-surface-3 transition-colors"
        >
          <IconCheck />
        </button>
      )}
    </li>
  );
}

export default function ChatSidebar({ sessions, activeId, onSelect, onNew, onRename, onDelete }) {
  return (
    <aside className="hidden md:flex w-[280px] shrink-0 flex-col bg-sidebar border-r border-white/5 h-screen">
      <div className="px-3 pt-5 pb-4">
        <div className="flex items-center gap-2 px-2 mb-5">
          <div className="w-6 h-6 rounded-md bg-gradient-to-br from-accent to-accent-light flex items-center justify-center shadow-lg shadow-accent/20">
            <span className="text-accent-ink font-display font-bold text-xs">R</span>
          </div>
          <span className="font-display font-semibold text-ink text-[15px] tracking-tight">
            Research Agent
          </span>
        </div>

        <button
          onClick={onNew}
          className="group w-full flex items-center justify-between gap-2 rounded-xl bg-gradient-to-b from-surface-3 to-surface-2 border border-white/[0.06] px-3.5 py-2.5 text-sm font-medium text-ink hover:from-accent hover:to-accent-light hover:text-white hover:border-transparent transition-all duration-200 ease-in-out shadow-sm"
        >
          <span className="flex items-center gap-2">
            <IconPlus />
            New research
          </span>
          <kbd className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-black/30 text-ink-faint group-hover:bg-white/15 group-hover:text-white/80 transition-colors">
            ⌘N
          </kbd>
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 pb-4">
        <p className="px-3 pt-2 pb-2 text-[11px] font-mono uppercase tracking-widest text-ink-faint">
          History
        </p>

        {sessions.length === 0 && (
          <p className="px-3 py-2 text-sm text-ink-faint leading-relaxed">
            Your research sessions will appear here.
          </p>
        )}

        <ul className="flex flex-col gap-0.5">
          {sessions.map((s) => (
            <SessionCard
              key={s.id}
              session={s}
              active={s.id === activeId}
              onSelect={onSelect}
              onRename={onRename}
              onDelete={onDelete}
            />
          ))}
        </ul>
      </nav>

      <div className="px-4 py-3 border-t border-white/5">
        <p className="text-[11px] font-mono text-ink-faint">v0.1.0 · local instance</p>
      </div>
    </aside>
  );
}