const NODE_LABELS = {
  planner: "Planning search strategy",
  retriever: "Searching sources",
  claim_extractor: "Extracting claims",
  verifier: "Verifying claims",
  synthesizer: "Writing answer",
};

export default function PipelineProgress({ nodeLog, status }) {
  if (nodeLog.length === 0 && status === "idle") return null;

  return (
    <ul className="space-y-2">
      {nodeLog.map((entry, i) => (
        <li key={i} className="flex items-center gap-2 text-sm text-neutral-600">
          <span className="h-2 w-2 rounded-full bg-green-500" />
          {NODE_LABELS[entry.node] ?? entry.node}
        </li>
      ))}
      {status === "running" && (
        <li className="flex items-center gap-2 text-sm text-neutral-400">
          <span className="h-2 w-2 animate-pulse rounded-full bg-neutral-400" />
          Working...
        </li>
      )}
    </ul>
  );
}