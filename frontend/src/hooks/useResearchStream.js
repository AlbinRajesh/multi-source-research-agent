import { useState, useCallback } from "react";
import { streamResearch } from "../lib/researchStream";

export function useResearchStream() {
  const [nodeLog, setNodeLog] = useState([]);
  const [report, setReport] = useState(null);
  const [citations, setCitations] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  const run = useCallback(async (topic) => {
    setNodeLog([]);
    setReport(null);
    setCitations([]);
    setError(null);
    setStatus("running");

    try {
      await streamResearch(topic, {
        onNode: (data) => {
          setNodeLog((log) => [...log, data]);
          if (data.node === "synthesize") {
            setReport(data.final_report);
            setCitations(data.citations ?? []);
          }
        },
        onDone: () => setStatus("done"),
        onError: (e) => {
          setError(e.message);
          setStatus("error");
        },
      });
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  }, []);

  return { run, nodeLog, report, citations, status, error };
}