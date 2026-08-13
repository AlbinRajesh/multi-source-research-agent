
import QueryInput from "./components/QueryInput";
import PipelineProgress from "./components/PipelineProgress";
import AnswerDisplay from "./components/AnswerDisplay";
import CitationList from "./components/CitationList";
import { useResearchStream } from "./hooks/useResearchStream";

export default function App() {
  const { run, nodeLog, report, citations, status, error } = useResearchStream();

  return (
    <div className="min-h-screen bg-background text-on-surface">
      <header className="border-b border-outline px-8 py-5">
        <h1 className="font-headline text-2xl">Research Agent</h1>
      </header>

      <main className="max-w-3xl mx-auto px-8 py-10 flex flex-col gap-8">
        <QueryInput onSubmit={run} disabled={status === "running"} />

        <PipelineProgress nodeLog={nodeLog} status={status} />

        {error && (
          <p className="text-red-400 bg-surface-container border border-outline rounded p-3 text-sm">
            {error}
          </p>
        )}

        <AnswerDisplay report={report} citations={citations} />
        <CitationList citations={citations} />
      </main>
    </div>
  );
}