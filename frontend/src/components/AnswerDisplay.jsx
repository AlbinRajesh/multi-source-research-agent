export default function AnswerDisplay({ report, citations }) {
  if (!report) return null;

  // Turn [1] [2] etc in the report text into clickable superscripts
  const parts = report.split(/(\[\d+\])/g);

  return (
    <article className="bg-surface-container border border-outline rounded p-8 md:p-10">
      <p className="font-body text-lg text-on-surface/90 leading-relaxed max-w-3xl whitespace-pre-wrap">
        {parts.map((part, i) => {
          const match = part.match(/^\[(\d+)\]$/);
          if (!match) return <span key={i}>{part}</span>;
          const idx = Number(match[1]);
          const citation = citations?.find((c) => c.index === idx);
          return (
            <sup key={i} className="text-primary cursor-pointer hover:underline ml-0.5" title={citation?.title}>
              [{idx}]
            </sup>
          );
        })}
      </p>
    </article>
  );
}