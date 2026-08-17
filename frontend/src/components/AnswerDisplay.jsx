import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function AnswerDisplay({ report, citations }) {
  if (!report) return null;

  // Convert bare [1] [2] markers into markdown links with a custom
  // "citation:" scheme, so ReactMarkdown parses them into <a> nodes
  // we can intercept and re-render as superscripts below. This has to
  // happen before ReactMarkdown sees the text — a plain text split
  // (like the old approach) can't coexist with real markdown parsing,
  // since headers/lists/etc need the full AST, not string chunks.
  const withCitationLinks = report.replace(
    /\[(\d+)\]/g,
    (match, num) => `[${match}](citation:${num})`
  );

  return (
    <article className="bg-surface-container border border-outline rounded p-8 md:p-10">
      <div className="font-body text-lg text-on-surface/90 leading-relaxed max-w-3xl prose prose-invert">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          urlTransform={(url) => url}   // Disable default sanitization so our citation: scheme survives
          components={{
            a: ({ href, children }) => {
              if (href?.startsWith("citation:")) {
                const idx = Number(href.replace("citation:", ""));
                const citation = citations?.find((c) => c.index === idx);
                return (
                  <sup
                    className="text-primary cursor-pointer hover:underline ml-0.5"
                    title={citation?.title}
                  >
                    [{idx}]
                  </sup>
                );
              }
              // real markdown links (shouldn't normally appear in
              // synthesizer output, but handle them safely)
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
    </article>
  );
}