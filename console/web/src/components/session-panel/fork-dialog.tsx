"use client";

import { useState } from "react";
import { GitBranch, X } from "lucide-react";

interface ForkDialogProps {
  onConfirm: (contextSummary: string) => void;
  onCancel: () => void;
}

export function ForkDialog({ onConfirm, onCancel }: ForkDialogProps) {
  const [summary, setSummary] = useState("");

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-300">
          <GitBranch className="w-3.5 h-3.5" />
          Fork Session
        </div>
        <button
          onClick={onCancel}
          className="p-0.5 rounded hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      <input
        type="text"
        value={summary}
        onChange={(e) => setSummary(e.target.value)}
        placeholder="Describe the fork context..."
        className="w-full px-2.5 py-1.5 rounded bg-zinc-800 border border-zinc-700 text-xs focus:outline-none focus:border-zinc-500"
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
        onClick={() => summary.trim() && onConfirm(summary.trim())}
        disabled={!summary.trim()}
        className="w-full py-1.5 rounded text-xs font-medium bg-blue-900/50 text-blue-400 hover:bg-blue-900/70 transition-colors disabled:opacity-30"
      >
        Create Fork
      </button>
    </div>
  );
}
