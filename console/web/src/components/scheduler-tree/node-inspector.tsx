"use client";

import { MonoLink, MonoText } from "@/components/mono-text";
import { SchedulerStatusBadge } from "@/components/scheduler-status-badge";
import { SectionCard } from "@/components/section-card";
import { EmptyStateMessage, ErrorStateMessage } from "@/components/state-message";
import { UserInputDetail } from "@/components/user-input-detail";
import type { AgentStateDetail, PendingEventItem, SchedulerTreeNode } from "@/lib/api";
import { formatLocalDateTime } from "@/lib/time";
import { formatWakeConditionSummary } from "@/lib/wake-condition";

type NodeInspectorProps = {
  node: SchedulerTreeNode | null;
  detail: AgentStateDetail | null;
  pendingEvents: PendingEventItem[];
  loading: boolean;
  error: string | null;
};

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-zinc-800/60 py-2 last:border-0">
      <span className="text-xs text-zinc-500">{label}</span>
      <div className="min-w-0 text-right text-sm text-zinc-200">{value}</div>
    </div>
  );
}

export function NodeInspector({
  node,
  detail,
  pendingEvents,
  loading,
  error,
}: NodeInspectorProps) {
  if (!node) {
    return (
      <SectionCard title="Inspector" bodyClassName="p-4">
        <EmptyStateMessage className="py-8 text-left text-zinc-500">
          Select a scheduler node to inspect its detail and pending events.
        </EmptyStateMessage>
      </SectionCard>
    );
  }

  return (
    <SectionCard title="Inspector" bodyClassName="p-4 space-y-4">
      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <MonoText className="text-sm text-zinc-100">{node.state_id}</MonoText>
          <SchedulerStatusBadge status={node.status} />
        </div>

        <div className="grid gap-2 sm:grid-cols-2">
          <InfoRow label="Root" value={<MonoText>{node.root_state_id}</MonoText>} />
          <InfoRow
            label="Session"
            value={
              node.session_id ? (
                <MonoLink href={`/sessions/${node.session_id}`}>{node.session_id}</MonoLink>
              ) : (
                "-"
              )
            }
          />
          <InfoRow label="Parent" value={node.parent_state_id || "-"} />
          <InfoRow label="Depth" value={node.depth} />
          <InfoRow label="Task ID" value={node.task_id || "-"} />
          <InfoRow
            label="Wake Condition"
            value={formatWakeConditionSummary(node.wake_condition)}
          />
          <InfoRow
            label="Created"
            value={formatLocalDateTime(node.created_at)}
          />
          <InfoRow
            label="Updated"
            value={formatLocalDateTime(node.updated_at)}
          />
        </div>
      </div>

      {detail?.task && (
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide text-zinc-500">Task</div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
            <UserInputDetail input={detail.task} />
          </div>
        </div>
      )}

      {(node.result_summary || node.last_error) && (
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wide text-zinc-500">
            Result / Error
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 text-sm text-zinc-200 whitespace-pre-wrap">
            {node.last_error ?? node.result_summary}
          </div>
        </div>
      )}

      <div className="space-y-2">
        <div className="text-xs uppercase tracking-wide text-zinc-500">
          Pending Events ({pendingEvents.length})
        </div>
        {loading ? (
          <div className="text-sm text-zinc-500">Loading selected node detail…</div>
        ) : pendingEvents.length === 0 ? (
          <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 text-sm text-zinc-500">
            No pending events.
          </div>
        ) : (
          <div className="space-y-2">
            {pendingEvents.map((event) => (
              <div
                key={event.id}
                className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm text-zinc-100">{event.event_type}</span>
                  <MonoText className="text-[11px] text-zinc-500">{event.id}</MonoText>
                </div>
                <pre className="mt-2 overflow-auto whitespace-pre-wrap text-xs text-zinc-400">
                  {JSON.stringify(event.payload, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </SectionCard>
  );
}
