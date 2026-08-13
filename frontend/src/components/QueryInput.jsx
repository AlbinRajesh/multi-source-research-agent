import { useState } from "react";

export default function QueryInput({ onSubmit, disabled }) {
  const [value, setValue] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (value.trim()) onSubmit(value.trim());
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-center bg-surface-container border border-outline rounded-lg p-1 focus-within:border-primary transition-colors"
    >
      <span className="material-symbols-outlined text-on-surface-variant pl-3 text-[20px]">
        search
      </span>
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask a research question..."
        className="flex-1 bg-transparent border-none text-on-surface placeholder:text-on-surface-variant/50 px-4 py-3 focus:outline-none focus:ring-0"
        disabled={disabled}
      />
      <button
        type="submit"
        disabled={disabled}
        className="bg-primary text-on-primary px-6 py-2.5 rounded font-medium text-sm hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed mr-1"
      >
        {disabled ? "Researching…" : "Research"}
      </button>
    </form>
  );
}