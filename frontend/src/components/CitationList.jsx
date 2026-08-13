const CONFIDENCE_STYLES = {
  verified: { label: "Verified", className: "bg-green-100 text-green-700" },
  corroborated: { label: "Corroborated", className: "bg-green-100 text-green-700" },
  single_source: { label: "Single source", className: "bg-amber-100 text-amber-700" },
  conflicting: { label: "Conflicting", className: "bg-red-100 text-red-700" },
};

export default function CitationList({ citations }) {
  if (!citations?.length) return null;

  return (
    <div className="mt-6 border-t border-neutral-200 pt-4">
      <h3 className="mb-2 text-sm font-medium text-neutral-700">Sources</h3>
      <ol className="space-y-2 text-sm">
        {citations.map((c) => {
          const confidence = CONFIDENCE_STYLES[c.confidence] ?? {
            label: c.confidence ?? "Unverified",
            className: "bg-neutral-100 text-neutral-600",
          };
          return (
            <li key={c.index ?? c.url} className="flex items-start gap-2">
              <span className="mt-0.5 text-neutral-400">[{c.index}]</span>
              <div className="min-w-0">
                <a
                  href={c.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-blue-600 hover:underline break-words"
                >
                  {c.title || c.url}
                </a>
                <div className="mt-1 flex items-center gap-2">
                  <span
                    className={`rounded px-1.5 py-0.5 text-xs font-medium ${confidence.className}`}
                  >
                    {confidence.label}
                  </span>
                  {c.source_type && (
                    <span className="text-xs text-neutral-400">{c.source_type}</span>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}