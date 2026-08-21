import PipelineRail from "./Pipelinerail";
import AnswerDisplay from "./AnswerDisplay";
import CitationList from "./CitationList";

export default function ChatMessage({ session }) {
  const { messages, nodeLog, status, error } = session;
  const lastIsUser = messages[messages.length - 1]?.role === "user";

  return (
    <div className="flex flex-col gap-4">
      {messages.map((m, i) =>
        m.role === "user" ? (
          <div key={i} className="flex justify-end">
            <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-gradient-to-br from-accent to-accent-light text-white px-4 py-2.5 text-sm leading-relaxed shadow-lg shadow-accent/10">
              {m.content}
            </div>
          </div>
        ) : (
          <div key={i} className="flex justify-start">
            <div className="max-w-[85%] w-full rounded-2xl rounded-tl-sm bg-surface border border-white/[0.06] px-5 py-4 shadow-xl shadow-black/20">
              <AnswerDisplay report={m.content} citations={m.citations ?? []} />
              <CitationList citations={m.citations ?? []} />
            </div>
          </div>
        )
      )}

      {/* In-flight assistant turn (pipeline log / error / spinner) */}
      {lastIsUser && (
        <div className="flex justify-start">
          <div className="max-w-[85%] w-full rounded-2xl rounded-tl-sm bg-surface border border-white/[0.06] px-5 py-4 shadow-xl shadow-black/20">
            <PipelineRail nodeLog={nodeLog} status={status} />

            {error && (
              <p className="mt-3 text-sm text-conflict bg-conflict-soft border border-conflict/20 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            {!error && status === "running" && nodeLog.length === 0 && (
              <p className="font-mono text-xs text-ink-faint">Starting…</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}