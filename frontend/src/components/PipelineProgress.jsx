const STAGES = ["plan", "search", "extract_claims", "verify", "synthesize"];
const STAGE_LABELS = {
  plan: "Plan",
  search: "Search",
  extract_claims: "Extract",
  verify: "Verify",
  synthesize: "Synthesize",
};

function stageSummary(entry) {
  if (!entry) return null;
  switch (entry.node) {
    case "plan":
      return `${entry.search_queries?.length ?? 0} queries planned`;
    case "search":
      return `${entry.result_count} sources retrieved`;
    case "extract_claims":
      return `${entry.claim_count} claims extracted`;
    case "verify":
      return `${entry.verified_count}/${entry.total_count} claims verified`;
    case "synthesize":
      return "Report synthesized";
    default:
      return null;
  }
}

export default function PipelineProgress({ nodeLog, status }) {
  if (nodeLog.length === 0 && status === "idle") return null;

  const completedNodes = new Set(nodeLog.map((e) => e.node));
  const activeIndex = STAGES.findIndex((s) => !completedNodes.has(s));
  const latest = nodeLog[nodeLog.length - 1];

  return (
    <div className="bg-surface-container border border-outline rounded p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2 flex-wrap">
        {STAGES.map((stage, i) => {
          const done = completedNodes.has(stage);
          const active = status === "running" && i === activeIndex;
          return (
            <div key={stage} className="flex items-center gap-2">
              <span
                className={`material-symbols-outlined text-[18px] ${
                  done
                    ? "text-primary"
                    : active
                    ? "text-on-surface animate-spin"
                    : "text-on-surface-variant"
                }`}
              >
                {done
                  ? "check_circle"
                  : active
                  ? "progress_activity"
                  : "radio_button_unchecked"}
              </span>
              <span
                className={`text-xs uppercase tracking-widest ${
                  done || active ? "text-on-surface" : "text-on-surface-variant"
                }`}
              >
                {STAGE_LABELS[stage]}
              </span>
              {i < STAGES.length - 1 && (
                <div className="w-6 h-px bg-outline ml-2" />
              )}
            </div>
          );
        })}
      </div>
      {latest && stageSummary(latest) && (
        <div className="text-sm font-mono text-on-surface-variant bg-background p-2 rounded">
          {stageSummary(latest)}
        </div>
      )}
    </div>
  );
}