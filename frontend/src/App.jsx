import QueryInput from "./components/QueryInput";
import PipelineProgress from "./components/PipelineProgress";
import AnswerDisplay from "./components/AnswerDisplay";
import CitationList from "./components/CitationList";
import { useResearchStream } from "./hooks/useResearchStream";

export default function App() {
  const { run, nodeLog, answer, citations, status, error } = useResearchStream();

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-8">
      <h1 className="text-2xl font-semibold">Research Agent</h1>
      <QueryInput onSubmit={run} disabled={status === "running"} />
      <PipelineProgress nodeLog={nodeLog} status={status} />
      {error && <p className="text-red-600">{error}</p>}
      <AnswerDisplay answer={answer} />
      <CitationList citations={citations} />
    </div>
  );
}