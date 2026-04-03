"use client";

import { useId, useState } from "react";

import { ErrorStateMessage } from "@/components/state-message";
import { cancelAgent, resumeAgent, steerAgent } from "@/lib/api";
import type { AgentStateDetail } from "@/lib/api";

type RootActionsProps = {
  state: AgentStateDetail;
  onActionComplete: () => Promise<void>;
};

const ACTIVE_ROOT_STATUSES = new Set(["pending", "running", "waiting", "queued"]);

export function RootActions({ state, onActionComplete }: RootActionsProps) {
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputId = useId();

  const canSteerOrCancel =
    state.parent_id === null && ACTIVE_ROOT_STATUSES.has(state.status);
  const canResume =
    state.parent_id === null &&
    state.is_persistent &&
    (state.status === "idle" || state.status === "failed");

  if (!canSteerOrCancel && !canResume) {
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

      {(canSteerOrCancel || canResume) && (
        <div className="space-y-3">
          <label htmlFor={inputId} className="sr-only">
            {canResume ? "Resume message" : "Steering message"}
          </label>
          <input
            id={inputId}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder={
              canResume ? "Resume message" : "Steer root agent with an operator hint"
            }
            className="ui-input"
          />

          <div className="flex flex-wrap gap-2">
            {canSteerOrCancel && (
              <button
                type="button"
                disabled={busy || !message.trim()}
                onClick={() =>
                  runAction(async () => {
                    await steerAgent(state.id, message.trim());
                  })
                }
                className="ui-button ui-button-primary"
              >
                Send Steering
              </button>
            )}

            {canResume && (
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
                Resume Root
              </button>
            )}

            {canSteerOrCancel && (
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
      )}
    </div>
  );
}
