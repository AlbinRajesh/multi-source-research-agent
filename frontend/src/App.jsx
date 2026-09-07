import { useEffect, useRef, useState } from "react";
import ChatSidebar from "./components/Chatsidebar";
import ChatMessage from "./components/ChatMessage";
import QueryInput from "./components/QueryInput";
import FileUploader from "./components/FileUploader";
import { IconShield } from "./components/icons";
import { useResearchStream } from "./hooks/useResearchStream";

export default function App() {
  const {
    sessions,
    activeSession,
    run,
    selectSession,
    newChat,
    renameSession,
    deleteSession,
    isRunning,
  } = useResearchStream();
  
  const scrollRef = useRef(null);
 
  // --- Document / RAG State ---
  const [docs, setDocs] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [showUploader, setShowUploader] = useState(false);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [activeSession?.nodeLog?.length, activeSession?.report]);

  // ⌘/Ctrl + Shift + N — new research
  useEffect(() => {
    const handler = (e) => {
      const isNewShortcut =
        (e.metaKey || e.ctrlKey) && (e.key === "n" || e.key === "N");
      if (isNewShortcut) {
        e.preventDefault();
        newChat();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [newChat]);

  // Pass sources array and selected doc IDs to run() based on uploaded docs
  const handleSubmit = (topic) => {
    const sources = docs.length > 0 ? ["web", "local"] : ["web"];
    run(topic, activeSession?.id, sources, selectedDocId ? [selectedDocId] : []);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-canvas">
       <ChatSidebar
        sessions={sessions}
        activeId={activeSession?.id}
        onSelect={selectSession}
        onNew={newChat}
        onRename={renameSession}
        onDelete={deleteSession}
        showUploader={showUploader}
        onToggleUploader={() => setShowUploader(!showUploader)}
        docCount={docs.length}
      />

      {/* Conditional File Uploader Drawer next to Sidebar */}
      {showUploader && (
        <div className="w-80 border-r border-white/[0.05] bg-surface-1 flex flex-col z-20 shadow-2xl">
          <div className="flex justify-between items-center p-4 border-b border-white/[0.05]">
            <h2 className="text-sm font-semibold text-ink">Local Documents</h2>
            <button
              onClick={() => setShowUploader(false)}
              className="text-ink-muted hover:text-ink transition"
              aria-label="Close uploader"
            >
              ✕
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <FileUploader
              docs={docs}
              selectedDocId={selectedDocId}
              onSelectDoc={setSelectedDocId}
              setDocs={setDocs}
            />
          </div>
        </div>
      )}

      <div className="flex-1 flex flex-col h-screen relative overflow-hidden">
        {/* Top bar control for Document Panel toggle */}
        

        <main className="flex-1 overflow-y-auto relative">
          {!activeSession && <AmbientGlow />}

          <div className="relative max-w-3xl mx-auto px-6 py-8 min-h-full flex flex-col">
            {!activeSession && (
              <div className="flex-1 flex items-center justify-center">
                <EmptyState onSubmit={handleSubmit} />
              </div>
            )}

            {activeSession && (
              <div className="flex flex-col gap-8">
                <ChatMessage session={activeSession} />
                <div ref={scrollRef} />
              </div>
            )}
          </div>
        </main>

        {activeSession && (
          <div className="border-t border-white/[0.05] bg-canvas/90 backdrop-blur-md px-6 py-4">
            <div className="max-w-3xl mx-auto">
              <QueryInput onSubmit={handleSubmit} disabled={isRunning} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function AmbientGlow() {
  return (
    <div
      className="pointer-events-none absolute inset-0 flex items-center justify-center overflow-hidden"
      aria-hidden="true"
    >
      <div
        className="w-[900px] h-[900px] rounded-full opacity-70"
        style={{
          background:
            "radial-gradient(circle, rgba(99,102,241,0.16) 0%, rgba(99,102,241,0.06) 35%, rgba(9,9,11,0) 70%)",
          animation: "glow-pulse 6s ease-in-out infinite",
        }}
      />
    </div>
  );
}

function EmptyState({ onSubmit }) {
  return (
    <div className="flex flex-col items-center text-center gap-7 w-full animate-fade-in-up">
      <style>{`
        @keyframes shimmerGradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-shimmer-text {
          background-size: 200% auto;
          animation: shimmerGradient 5s ease infinite;
        }
        .animate-fade-in-up {
          animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
      `}</style>

      <div className="flex flex-col items-center gap-4 relative group">
        {/* Soft Ambient Backlight Glow behind the heading */}
        <div className="absolute -inset-1 blur-xl bg-gradient-to-r from-[#6366f1]/30 via-indigo-500/10 to-transparent rounded-full opacity-70 group-hover:opacity-100 transition duration-1000 group-hover:duration-200 pointer-events-none" />

        {/* Masterpiece Heading */}
        <h2 className="relative font-display font-bold text-4xl sm:text-6xl tracking-tight text-center mb-2 bg-gradient-to-r from-white via-[#c7d2fe] to-[#6366f1] bg-clip-text text-transparent animate-shimmer-text drop-shadow-[0_4px_24px_rgba(99,102,241,0.25)]">
          Explore a topic?
        </h2>

        <span className="inline-flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-surface-2 px-3.5 py-1.5 text-xs text-ink-muted shadow-sm">
          <IconShield className="text-verified" />
          Every claim checked against its source before it reaches you
        </span>

      </div>

      <div className="w-full max-w-xl">
        <QueryInput onSubmit={onSubmit} disabled={false} />
      </div>
    </div>
  );
}