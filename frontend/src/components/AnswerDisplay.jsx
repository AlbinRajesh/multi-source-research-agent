// AnswerDisplay.jsx
export default function AnswerDisplay({ answer }) {
  if (!answer) return null;
  return (
    <div className="prose max-w-none rounded-lg border border-neutral-200 p-4">
      <p>{answer}</p>
    </div>
  );
}