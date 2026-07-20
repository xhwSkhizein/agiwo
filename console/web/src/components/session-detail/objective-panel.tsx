"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { listSessionObjectives, type ObjectiveView } from "@/lib/api";
import {
  EmptyStateMessage,
  ErrorStateMessage,
  TextStateMessage,
} from "@/components/state-message";

type ObjectivePanelProps = {
  sessionId: string;
};

export function ObjectivePanel({ sessionId }: ObjectivePanelProps) {
  const [objectives, setObjectives] = useState<ObjectiveView[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedTimeline, setExpandedTimeline] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const items = await listSessionObjectives(sessionId);
      setObjectives(items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load objectives");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) {
    return <TextStateMessage>Loading objectives…</TextStateMessage>;
  }
  if (error) {
    return <ErrorStateMessage>{error}</ErrorStateMessage>;
  }
  if (objectives.length === 0) {
    return (
      <EmptyStateMessage>No objectives in this session yet.</EmptyStateMessage>
    );
  }

  const active =
    objectives.find((item) => !item.is_terminal) ?? objectives[objectives.length - 1];
  const delivered = Boolean(active.delivery_report);

  return (
    <section className="space-y-4 rounded-xl border border-line bg-panel/40 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="ui-section-kicker">Objective</p>
          <h2 className="text-lg font-semibold text-ink">{active.status}</h2>
          <p className="mt-1 break-all font-mono text-xs text-ink-muted">
            {active.objective_id}
          </p>
        </div>
        <button
          type="button"
          className="ui-button ui-button-ghost text-xs"
          onClick={() => void load()}
        >
          Refresh
        </button>
      </div>

      {delivered ? (
        <div className="space-y-3">
          <div>
            <p className="ui-section-kicker">Delivery</p>
            <div className="mt-2 whitespace-pre-wrap text-sm text-ink">
              {active.delivery_report}
            </div>
          </div>
          {active.artifacts.length > 0 && (
            <div>
              <p className="ui-section-kicker">Artifacts</p>
              <ul className="mt-2 space-y-1 text-sm">
                {active.artifacts.map((artifact) => (
                  <li key={artifact.artifact_id} className="break-all text-ink-muted">
                    {artifact.path}
                    {artifact.summary ? ` — ${artifact.summary}` : ""}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <button
            type="button"
            className="ui-button ui-button-ghost text-xs"
            onClick={() => setExpandedTimeline((value) => !value)}
          >
            {expandedTimeline ? "Hide timeline" : "Show timeline"}
          </button>
        </div>
      ) : (
        <div className="space-y-2 text-sm text-ink-muted">
          <p>
            Budget remaining: handoffs {active.budget.handoffs.remaining}/
            {active.budget.handoffs.limit}, verification{" "}
            {active.budget.verification_attempts.remaining}/
            {active.budget.verification_attempts.limit}
          </p>
        </div>
      )}

      {(!delivered || expandedTimeline) && (
        <ol className="space-y-2 border-t border-line pt-3">
          {active.timeline.map((node) => (
            <li key={node.fact_id} className="text-sm">
              <div className="flex flex-wrap items-baseline gap-2">
                <span className="font-mono text-xs text-ink-muted">
                  #{node.sequence}
                </span>
                <span className="font-medium text-ink">{node.kind}</span>
              </div>
              <p className="mt-0.5 text-ink-muted">{node.summary}</p>
              {typeof node.refs.run_id === "string" && (
                <Link
                  href={`/traces?run_id=${encodeURIComponent(String(node.refs.run_id))}`}
                  className="text-xs text-accent underline-offset-2 hover:underline"
                >
                  Open run {String(node.refs.run_id).slice(0, 12)}
                </Link>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
