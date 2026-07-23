"use client";

import { useId, useState } from "react";

import { ErrorStateMessage } from "@/components/state-message";
import { cancelAgent, resumeAgent } from "@/lib/api";
import type { AgentStateDetail } from "@/lib/api";

type RootActionsProps = {
  state: AgentStateDetail;
  onActionComplete: () => Promise<void>;
};

const ACTIVE_ROOT_STATUSES = new Set(["pending", "running", "waiting", "queued"]);

/**
 * Root-only orchestration controls for a selected AgentStateDetail.
 *
 * Enqueue uses Scheduler.enqueue_input via the resume API (idle/failed next cycle,
 * or live/USER_HINT while active). Cancel remains separate.
 */
export function RootActions({ state, onActionComplete }: RootActionsProps) {
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputId = useId();

  const canCancel =
    state.parent_id === null && ACTIVE_ROOT_STATUSES.has(state.status);
  const canEnqueue =
    state.parent_id === null &&
    state.is_persistent &&
    (state.status === "idle" ||
      state.status === "failed" ||
      ACTIVE_ROOT_STATUSES.has(state.status));

  if (!canCancel && !canEnqueue) {
    return null;
  }

  async function runAction(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      setMessage("");
      await onActionComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-2xl border border-line bg-panel p-4 space-y-3">
      <div>
        <h3 className="text-sm font-medium">Root Controls</h3>
        <p className="mt-1 text-xs text-ink-muted">
          Root-only orchestration controls for the selected scheduler state.
        </p>
      </div>

      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      <div className="space-y-3">
        {canEnqueue && (
          <>
            <label htmlFor={inputId} className="sr-only">
              Enqueue message
            </label>
            <input
              id={inputId}
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Enqueue input for this persistent root"
              className="ui-input"
            />
          </>
        )}

        <div className="flex flex-wrap gap-2">
          {canEnqueue && (
            <button
              type="button"
              disabled={busy || !message.trim()}
              onClick={() =>
                runAction(async () => {
                  await resumeAgent(state.id, message.trim());
                })
              }
              className="ui-button ui-button-primary"
            >
              Enqueue Input
            </button>
          )}

          {canCancel && (
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                runAction(async () => {
                  await cancelAgent(state.id, "Cancelled by operator");
                })
              }
              className="ui-button ui-button-danger"
            >
              Cancel Root
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
