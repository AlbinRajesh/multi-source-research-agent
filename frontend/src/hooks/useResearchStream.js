import { useCallback, useMemo, useState } from "react";
import { streamResearch } from "../lib/researchStream";

function newSession(topic) {
  const id =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    id,
    threadId: id, // reused as the backend thread_id — lets a session resume later
    messages: [{ role: "user", content: topic }],
    nodeLog: [],
    status: "running", // running | done | error
    error: null,
    createdAt: Date.now(),
  };
}

export function useResearchStream() {
  const [sessions, setSessions] = useState([]);
  const [activeId, setActiveId] = useState(null);

  const patchSession = useCallback((id, patch) => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === id
          ? { ...s, ...(typeof patch === "function" ? patch(s) : patch) }
          : s
      )
    );
  }, []);

  const run = useCallback(
    async (topic, existingSessionId, sources) => {
      let session;
      if (existingSessionId) {
        session = sessions.find((s) => s.id === existingSessionId);
      }
      if (!session) {
        session = newSession(topic);
        setSessions((prev) => [session, ...prev]);
        setActiveId(session.id);
      } else {
        patchSession(session.id, (s) => ({
          messages: [...s.messages, { role: "user", content: topic }],
          status: "running",
          nodeLog: [],
          error: null,
        }));
      }

      try {
        await streamResearch(topic, session.threadId, sources, {
          onNode: (data) => {
            patchSession(session.id, (s) => {
              const nodeLog = [...s.nodeLog, data];

              if (data.error) {
                return { nodeLog, error: data.error, status: "error" };
              }

              if (data.node === "route" && data.is_casual) {
                return {
                  nodeLog,
                  status: "done",
                  messages: [
                    ...s.messages,
                    { role: "assistant", content: data.final_report, citations: [] },
                  ],
                };
              }

              if (data.node === "fast_local_answer") {
                if (data.error) {
                  return { nodeLog, error: data.error, status: "error" };
                }
                if (data.route_decision === "escalate") {
                  // fell through to full pipeline — don't render a message yet,
                  // wait for the eventual "synthesize" event instead
                  return { nodeLog };
                }
                return {
                  nodeLog,
                  messages: [
                    ...s.messages,
                    {
                      role: "assistant",
                      content: data.final_report,
                      citations: data.citations ?? [],
                    },
                  ],
                };
              }

              if (data.node === "synthesize") {
                if (data.error) {
                  return { nodeLog, error: data.error, status: "error" };
                }
                return {
                  nodeLog,
                  messages: [
                    ...s.messages,
                    {
                      role: "assistant",
                      content: data.final_report,
                      citations: data.citations ?? [],
                    },
                  ],
                };
              }
              return { nodeLog };
            });
          },
          onDone: () => patchSession(session.id, { status: "done" }),
          onError: (e) =>
            patchSession(session.id, { error: e.message, status: "error" }),
        });
      } catch (e) {
        patchSession(session.id, { error: e.message, status: "error" });
      }
    },
    [sessions, patchSession]
  );

  const selectSession = useCallback((id) => setActiveId(id), []);

  const newChat = useCallback(() => setActiveId(null), []);

  const renameSession = useCallback((id, topic) => {
    patchSession(id, { topic });
  }, [patchSession]);

  const deleteSession = useCallback((id) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    setActiveId((current) => (current === id ? null : current));
  }, []);

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeId) ?? null,
    [sessions, activeId]
  );

  const isRunning = activeSession?.status === "running";

  return {
    sessions,
    activeSession,
    run,
    selectSession,
    newChat,
    renameSession,
    deleteSession,
    isRunning,
  };
}