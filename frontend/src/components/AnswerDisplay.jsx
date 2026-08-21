import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function AnswerDisplay({ report, citations }) {
  if (!report) return null;

  const withCitationLinks = report.replace(
    /\[(\d+)\]/g,
    (match, num) => `[${match}](citation:${num})`
  );

  return (
    <div className="font-body text-[15px] text-ink/90 leading-[1.75] max-w-none prose prose-invert prose-headings:font-display prose-headings:font-semibold prose-headings:tracking-tight prose-headings:text-ink prose-a:text-accent-light prose-strong:text-ink prose-p:my-3 prose-li:my-1 prose-hr:border-border">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        urlTransform={(url) => url}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith("citation:")) {
              const idx = Number(href.replace("citation:", ""));
              const citation = citations?.find((c) => c.index === idx);
              return (
                <sup
                  className="font-mono text-[11px] text-accent-light bg-accent-soft rounded px-1 py-0.5 cursor-default ml-0.5 not-italic"
                  title={citation?.title}
                >
                  {idx}
                </sup>
              );
            }
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {withCitationLinks}
      </ReactMarkdown>
    </div>
  );
}