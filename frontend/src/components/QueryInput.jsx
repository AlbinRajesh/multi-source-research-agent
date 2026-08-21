import { useState } from "react";
import { IconSend, IconLoader } from "./icons";

export default function QueryInput({ onSubmit, disabled }) {
  const [value, setValue] = useState("");

  const submit = () => {
    if (value.trim() && !disabled) {
      onSubmit(value.trim());
      setValue("");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col items-center">


      {/* Command bar */}
      <div className="w-full relative rounded-2xl bg-surface-2 border border-white/[0.06] shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] focus-within:ring-2 focus-within:ring-[#6366f1]/50 focus-within:border-transparent transition-all duration-200 ease-in-out">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a research question…"
          rows={1}
          disabled={disabled}
          className="w-full resize-none bg-transparent border-none text-[15px] text-ink placeholder:text-ink-faint px-4 pt-3.5 pb-8 focus:outline-none disabled:opacity-50"
        />

        <div className="flex items-center justify-between px-4 pb-2.5">
          <span className="font-mono text-[10px] text-ink-faint tracking-wide">
            <kbd className="text-ink-muted">↵</kbd> to send · <kbd className="text-ink-muted">⇧↵</kbd> for new line
          </span>

          <button
            type="button"
            onClick={submit}
            disabled={disabled || !value.trim()}
            className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-[#6366f1] to-indigo-400 text-white shadow-lg shadow-[#6366f1]/25 hover:shadow-[#6366f1]/40 hover:scale-[1.03] active:scale-95 transition-all duration-200 ease-in-out disabled:opacity-30 disabled:shadow-none disabled:hover:scale-100 disabled:cursor-not-allowed shrink-0"
            aria-label="Send"
          >
            {disabled ? <IconLoader /> : <IconSend />}
          </button>
        </div>
      </div>
    </div>
  );
}