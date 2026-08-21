import { IconExternal } from "./icons";

const CONFIDENCE_STYLES = {
  verified: { label: "Verified", dot: "bg-verified", text: "text-verified", bg: "bg-verified-soft" },
  corroborated: { label: "Verified", dot: "bg-verified", text: "text-verified", bg: "bg-verified-soft" },
  single_source: { label: "Single source", dot: "bg-single", text: "text-single", bg: "bg-single-soft" },
  conflicting: { label: "Conflicting", dot: "bg-conflict", text: "text-conflict", bg: "bg-conflict-soft" },
};

export default function CitationList({ citations }) {
  if (!citations?.length) return null;

  return (
    <div className="mt-6 pt-5 border-t border-border">
      <p className="mb-3 text-[11px] font-mono uppercase tracking-widest text-ink-faint">
        Sources · {citations.length}
      </p>
      <ul className="grid sm:grid-cols-2 gap-2">
        {citations.map((c) => {
          const conf =
            CONFIDENCE_STYLES[c.confidence] ?? {
              label: c.confidence ?? "Unverified",
              dot: "bg-unconfirmed",
              text: "text-ink-muted",
              bg: "bg-unconfirmed-soft",
            };
          return (
            <li key={c.index ?? c.url}>
              <a
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group flex flex-col gap-1.5 h-full rounded-lg border border-white/[0.06] bg-surface-2 px-3 py-2.5 hover:border-accent/40 hover:bg-surface-3 transition-all duration-200 ease-in-out"
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="font-mono text-[11px] text-ink-faint mt-0.5">[{c.index}]</span>
                  <IconExternal className="text-ink-faint group-hover:text-accent-light transition-colors shrink-0 mt-0.5" />
                </div>
                <span className="text-sm text-ink/90 leading-snug line-clamp-2">
                  {c.title || c.url}
                </span>
                <span
                  className={`inline-flex items-center gap-1.5 self-start rounded-full px-2 py-0.5 text-[11px] font-medium ${conf.bg} ${conf.text}`}
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${conf.dot}`} />
                  {conf.label}
                </span>
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}