"use client";

import { useId, useState } from "react";
import { GitBranch, X } from "lucide-react";

interface ForkDialogProps {
  onConfirm: (contextSummary: string) => void;
  onCancel: () => void;
}

/**
 * Renders a dialog that collects a fork summary and invokes confirm or cancel callbacks.
 *
 * The dialog associates its label with the text input for accessibility, autofocuses the input,
 * disables submission when the trimmed summary is empty, and supports keyboard actions:
 * pressing Enter submits when the trimmed summary is non-empty, and pressing Escape cancels.
 *
 * @param onConfirm - Called with the trimmed summary when the user confirms creation
 * @param onCancel - Called when the user cancels or closes the dialog
 * @returns The fork session dialog element
 */
export function ForkDialog({ onConfirm, onCancel }: ForkDialogProps) {
  const [summary, setSummary] = useState("");
  const inputId = useId();

  return (
    <section
      aria-label="Fork session"
      className="space-y-2 rounded-2xl border border-line bg-panel p-3"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs font-medium text-ink-soft">
          <GitBranch className="w-3.5 h-3.5" />
          Fork Session
        </div>
        <button
          type="button"
          onClick={onCancel}
          aria-label="Close fork form"
          className="ui-button ui-button-ghost ui-button-icon min-h-8 min-w-8 rounded-md p-1"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      <label htmlFor={inputId} className="ui-field-label mb-0">
        Fork summary
      </label>
      <input
        id={inputId}
        type="text"
        value={summary}
        onChange={(e) => setSummary(e.target.value)}
        placeholder="Describe the fork context..."
        className="ui-input"
        onKeyDown={(e) => {
          if (e.key === "Enter" && summary.trim()) {
            onConfirm(summary.trim());
          }
          if (e.key === "Escape") {
            onCancel();
          }
        }}
        autoFocus
      />
      <button
        type="button"
        onClick={() => summary.trim() && onConfirm(summary.trim())}
        disabled={!summary.trim()}
        className="ui-button ui-button-primary w-full"
      >
        Create Fork
      </button>
    </section>
  );
}
