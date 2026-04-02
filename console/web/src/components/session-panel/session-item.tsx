"use client";

import { GitBranch, Workflow } from "lucide-react";
import type { ChatSessionItem } from "@/lib/api";
import { UserInputCompact } from "@/components/user-input-detail";

function formatRelativeTime(dateStr: string | null): string {
  if (!dateStr) return "";
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

interface SessionItemProps {
  session: ChatSessionItem;
  isCurrent: boolean;
  onSwitch: () => void;
  onFork: () => void;
}

export function SessionItem({
  session,
  isCurrent,
  onSwitch,
  onFork,
}: SessionItemProps) {
  const schedulerActive = Boolean(session.root_state_status);

  return (
    <div
      className={`rounded-lg border p-3 transition-colors ${
        isCurrent
          ? "border-blue-700 bg-blue-950/30"
          : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 cursor-pointer"
      }`}
      onClick={isCurrent ? undefined : onSwitch}
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-zinc-300">
            {session.session_id.slice(0, 8)}
          </span>
          {isCurrent && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-900/50 text-blue-300 font-medium">
              current
            </span>
          )}
          {schedulerActive && (
            <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-300">
              <Workflow className="w-3 h-3" />
              {session.root_state_status}
            </span>
          )}
        </div>
        {isCurrent && (
          <button
            onClick={(event) => {
              event.stopPropagation();
              onFork();
            }}
            className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors px-1.5 py-0.5 rounded hover:bg-zinc-800"
            title="Fork session"
          >
            <GitBranch className="w-3 h-3" />
            Fork
          </button>
        )}
      </div>

      {session.last_user_input && (
        <div className="mb-2 text-xs">
          <UserInputCompact input={session.last_user_input} maxLength={90} />
        </div>
      )}

      {session.source_session_id && (
        <div className="flex items-center gap-1.5 text-[11px] text-zinc-500 mb-1">
          <GitBranch className="w-3 h-3" />
          <span>from</span>
          <span className="font-mono">{session.source_session_id.slice(0, 8)}</span>
        </div>
      )}

      <div className="flex items-center justify-between text-[11px] text-zinc-600">
        <span>
          {session.run_count} runs / {session.step_count} steps
        </span>
        <span>{formatRelativeTime(session.updated_at)}</span>
      </div>

      {session.fork_context_summary && (
        <div
          className="mt-1.5 text-[10px] text-zinc-600 truncate"
          title={session.fork_context_summary}
        >
          {session.fork_context_summary}
        </div>
      )}
    </div>
  );
}
