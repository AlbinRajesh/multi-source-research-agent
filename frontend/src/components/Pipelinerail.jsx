const STAGES = ["plan", "search", "extract_claims", "verify", "synthesize"];

const STAGE_META = {
  plan: { label: "Planning", icon: "ti-route" },
  search: { label: "Searching", icon: "ti-search" },
  extract_claims: { label: "Extracting claims", icon: "ti-quote" },
  verify: { label: "Verifying", icon: "ti-shield-check" },
  synthesize: { label: "Synthesizing", icon: "ti-file-text" },
};

function stageSummary(entry) {
  if (!entry) return null;
  switch (entry.node) {
    case "plan":
      return `${entry.search_queries?.length ?? 0} queries planned`;
    case "search":
      return entry.local_count
        ? `${entry.result_count} sources retrieved (${entry.local_count} from your docs)`
        : `${entry.result_count} sources retrieved`;
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

export default function PipelineRail({ nodeLog, status }) {
  if (!nodeLog || nodeLog.length === 0) return null;

  const isCasual = nodeLog.some((e) => e.node === "route" && e.is_casual);
  if (isCasual) return null;

  const completedNodes = new Set(nodeLog.map((e) => e.node));
  const activeIndex = STAGES.findIndex((s) => !completedNodes.has(s));
  const finished = status !== "running";
  const progressCount = finished ? STAGES.length : Math.max(activeIndex, 0);
  const progressPct = (progressCount / (STAGES.length - 1)) * 100;

  return (
    <div className="flex flex-col gap-3 py-1">
      <div className="relative flex items-center justify-between">
        <div className="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-[2px] bg-surface-3 rounded-full" />
        <div
          className="absolute left-0 top-1/2 -translate-y-1/2 h-[2px] bg-accent-light rounded-full transition-all duration-500 ease-out"
          style={{ width: `${Math.min(progressPct, 100)}%` }}
        />
        {STAGES.map((stage, i) => {
          const done = completedNodes.has(stage);
          const active = status === "running" && i === activeIndex;
          return (
            <div
              key={stage}
              className={`relative z-10 flex items-center justify-center w-6 h-6 rounded-full border transition-all duration-300 ${
                done
                  ? "bg-accent-light border-accent-light text-white"
                  : active
                  ? "bg-surface border-accent-light text-accent-light animate-pulse"
                  : "bg-surface-2 border-white/10 text-ink-faint"
              }`}
              title={STAGE_META[stage].label}
            >
              <i className={`ti ${STAGE_META[stage].icon} text-[13px]`} aria-hidden="true" />
            </div>
          );
        })}
      </div>

      {!finished && (
        <p className="font-mono text-xs text-ink-muted flex items-center gap-1.5">
          <i className={`ti ${STAGE_META[nodeLog[nodeLog.length - 1].node]?.icon ?? "ti-loader-2"} animate-spin text-[12px]`} aria-hidden="true" />
          {STAGE_META[nodeLog[nodeLog.length - 1].node]?.label ?? "Working"}
          {stageSummary(nodeLog[nodeLog.length - 1]) ? ` — ${stageSummary(nodeLog[nodeLog.length - 1])}` : "…"}
        </p>
      )}
      {finished && (
        <p className="font-mono text-xs text-ink-faint flex items-center gap-1.5">
          <i className="ti ti-check text-verified text-[12px]" aria-hidden="true" />
          {nodeLog.length} stages · complete
        </p>
      )}
    </div>
  );
}