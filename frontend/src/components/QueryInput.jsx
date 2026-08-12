import { useState } from "react";

export default function QueryInput({ onSubmit, disabled }) {
  const [value, setValue] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (value.trim()) onSubmit(value.trim());
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask a research question..."
        className="flex-1 rounded-lg border border-neutral-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-neutral-900"
        disabled={disabled}
      />
      <button
        type="submit"
        disabled={disabled}
        className="rounded-lg bg-neutral-900 px-4 py-2 text-white disabled:opacity-50"
      >
        Research
      </button>
    </form>
  );
}