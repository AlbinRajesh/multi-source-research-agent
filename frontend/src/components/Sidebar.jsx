export default function Sidebar({ threads, activeThreadId, onSelectThread, onNewThread }) {
  return (
    <aside className="fixed left-0 top-0 h-screen w-72 bg-surface border-r border-outline flex flex-col z-50">
      <div className="p-6 flex flex-col gap-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
            <span className="material-symbols-outlined text-on-primary">hub</span>
          </div>
          <span className="font-headline text-lg">AGENT CORE</span>
        </div>
        <button onClick={onNewThread} className="flex items-center justify-center gap-2 w-full py-3 px-4 bg-primary text-on-primary rounded text-sm font-medium hover:opacity-90 transition-opacity">
          <span className="material-symbols-outlined text-[18px]">add</span>New Thread
        </button>
      </div>
      <div className="px-4 mb-2 text-xs uppercase tracking-widest text-on-surface-variant opacity-50">Recent Research</div>
      <nav className="flex-1 overflow-y-auto px-2 space-y-1">
        {threads.map((t) => (
          <button
            key={t.id}
            onClick={() => onSelectThread(t.id)}
            className={`w-full text-left flex items-center px-4 py-3 rounded-lg transition-all ${
              t.id === activeThreadId ? "bg-surface-container-high text-on-surface border-l-2 border-primary" : "text-on-surface-variant hover:bg-surface-container"
            }`}
          >
            <span className="material-symbols-outlined mr-3 text-[20px]">analytics</span>
            <span className="truncate">{t.topic}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}