import { useState, useCallback } from "react";
import { streamResearch } from "../lib/researchStream";

export function useResearchStream() {
  const [nodeLog, setNodeLog] = useState([]);
  const [answer, setAnswer] = useState(null);
  const [citations, setCitations] = useState([]);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [error, setError] = useState(null);

  const run = useCallback(async (query) => {
    setNodeLog([]);
    setAnswer(null);
    setCitations([]);
    setError(null);
    setStatus("running");

    await streamResearch(query, {
      onNode: ({ node, output }) => {
        setNodeLog((log) => [...log, { node, output, ts: Date.now() }]);
        if (node === "synthesizer" && output.final_answer) {
          setAnswer(output.final_answer);
          setCitations(output.citations ?? []);
        }
      },
      onDone: () => setStatus("done"),
      onError: (e) => {
        setError(e.message);
        setStatus("error");
      },
    });
  }, []);

  return { run, nodeLog, answer, citations, status, error };
}