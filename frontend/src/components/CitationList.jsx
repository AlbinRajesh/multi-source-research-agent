// CitationList.jsx
export default function CitationList({ citations }) {
  if (!citations?.length) return null;
  return (
    <ul className="mt-3 space-y-1 text-sm text-neutral-500">
      {citations.map((c, i) => (
        <li key={i}>
          [{i + 1}]{" "}
          <a href={c.url} target="_blank" rel="noreferrer" className="underline">
            {c.title ?? c.url}
          </a>
        </li>
      ))}
    </ul>
  );
}