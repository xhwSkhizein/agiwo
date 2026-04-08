"use client";

import { GitBranch, Trash2, Workflow } from "lucide-react";
import type { ChatSessionItem } from "@/lib/api";
import { UserInputCompact } from "@/components/user-input-detail";
import { cn } from "@/lib/utils";

/**
 * Format a timestamp string into a concise human-readable relative time.
 *
 * @param dateStr - A parseable date/time string or `null`; if falsy the function returns an empty string.
 * @returns `""` for falsy input; `"just now"` for < 1 minute; `"{N}m ago"` for minutes < 60; `"{N}h ago"` for hours < 24; `"{N}d ago"` otherwise.
 */
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
  onDelete?: () => void;
}

/**
 * Renders a session list item showing session metadata, status badges, and actions.
 *
 * @param session - Session data to display (id, counts, timestamps, status, and optional fork/source info)
 * @param isCurrent - Whether this session is the currently active session
 * @param onSwitch - Callback invoked when a non-current session item is clicked to switch sessions
 * @param onFork - Callback invoked when the Fork control is activated for the current session
 * @returns The rendered session item element
 */
export function SessionItem({
  session,
  isCurrent,
  onSwitch,
  onFork,
  onDelete,
}: SessionItemProps) {
  const schedulerActive = Boolean(session.root_state_status);
  const containerClassName = cn(
    "rounded-2xl border p-3 transition-colors",
    isCurrent
      ? "border-accent bg-panel-strong"
      : "border-line bg-panel hover:border-line-strong",
  );

  const content = (
    <>
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-xs font-mono text-ink-soft">
            {session.session_id.slice(0, 8)}
          </span>
          {isCurrent && (
            <span className="rounded-full bg-accent/15 px-1.5 py-0.5 text-[10px] font-medium text-accent">
              current
            </span>
          )}
          {schedulerActive && (
            <span className="inline-flex items-center gap-1 rounded-full bg-success/15 px-1.5 py-0.5 text-[10px] text-success">
              <Workflow className="w-3 h-3" />
              {session.root_state_status}
            </span>
          )}
        </div>
        {isCurrent && (
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={onFork}
              className="ui-button ui-button-ghost min-h-7 rounded-full px-2 py-1 text-[10px]"
              aria-label={`Fork session ${session.session_id.slice(0, 8)}`}
            >
              <GitBranch className="w-3 h-3" />
              Fork
            </button>
            {onDelete && (
              <button
                type="button"
                onClick={onDelete}
                className="ui-button ui-button-ghost min-h-7 rounded-full px-2 py-1 text-[10px] text-danger hover:text-danger"
                aria-label={`Delete session ${session.session_id.slice(0, 8)}`}
              >
                <Trash2 className="w-3 h-3" />
              </button>
            )}
          </div>
        )}
      </div>

      {session.last_user_input && (
        <div className="mb-2 text-xs">
          <UserInputCompact input={session.last_user_input} maxLength={90} />
        </div>
      )}

      {session.source_session_id && (
        <div className="mb-1 flex items-center gap-1.5 text-[11px] text-ink-muted">
          <GitBranch className="w-3 h-3" />
          <span>from</span>
          <span className="font-mono">{session.source_session_id.slice(0, 8)}</span>
        </div>
      )}

      <div className="flex items-center justify-between text-[11px] text-ink-faint">
        <span>
          {session.run_count} runs / {session.step_count} steps
        </span>
        <span>{formatRelativeTime(session.updated_at)}</span>
      </div>

      {session.fork_context_summary && (
        <div
          className="mt-1.5 truncate text-[10px] text-ink-faint"
          title={session.fork_context_summary}
        >
          {session.fork_context_summary}
        </div>
      )}
    </>
  );

  return (
    <div className={containerClassName}>
      {isCurrent ? (
        content
      ) : (
        <button
          type="button"
          onClick={onSwitch}
          className="block w-full text-left"
          aria-label={`Switch to session ${session.session_id.slice(0, 8)}`}
        >
          {content}
        </button>
      )}
    </div>
  );
}
