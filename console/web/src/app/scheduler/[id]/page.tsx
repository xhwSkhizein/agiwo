"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { BackHeader } from "@/components/back-header";
import { JsonDisclosure } from "@/components/json-disclosure";
import { MetricCard } from "@/components/metric-card";
import { MonoLink, MonoText } from "@/components/mono-text";
import { PillBadge } from "@/components/pill-badge";
import { SectionCard } from "@/components/section-card";
import { SchedulerStatusBadge } from "@/components/scheduler-status-badge";
import { ErrorStateMessage, FullPageMessage } from "@/components/state-message";
import { TokenSummaryCards } from "@/components/token-summary-cards";
import {
  getAgentState,
  getAgentStateChildren,
  getPendingEvents,
  steerAgent,
  cancelAgent,
  resumeAgent,
  getAgent,
} from "@/lib/api";
import { UserInputDetail, UserInputCompact } from "@/components/user-input-detail";
import type {
  AgentConfig,
  AgentStateDetail,
  AgentStateListItem,
  ContentPartPayload,
  PendingEventItem,
  UserInput,
} from "@/lib/api";
import {
  formatDurationMs,
  formatTokenCount,
  formatUsd,
  normalizeRunMetricsSummary,
} from "@/lib/metrics";
import { formatLocalDateTime } from "@/lib/time";
import {
  formatWakeConditionTimer,
  getWaitsetProgress,
} from "@/lib/wake-condition";
import { usePageVisibility } from "@/hooks/use-page-visibility";

function InfoRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-4 border-b border-line py-2.5 last:border-0">
      <span className="w-36 shrink-0 pt-0.5 text-xs text-ink-faint">{label}</span>
      <div className="min-w-0 text-sm">{children}</div>
    </div>
  );
}

function extractTaskContext(input: UserInput | null | undefined): {
  source: string | null;
  metadata: Record<string, unknown>;
} {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    return { source: null, metadata: {} };
  }
  const maybeContext = "context" in input ? input.context : null;
  if (!maybeContext || typeof maybeContext !== "object") {
    return { source: null, metadata: {} };
  }
  return {
    source: typeof maybeContext.source === "string" ? maybeContext.source : null,
    metadata:
      maybeContext.metadata && typeof maybeContext.metadata === "object"
        ? maybeContext.metadata
        : {},
  };
}

function extractTaskParts(input: UserInput | null | undefined): ContentPartPayload[] {
  if (!input) return [];
  if (Array.isArray(input)) return input;
  if (typeof input !== "object") return [];
  if ("content" in input && Array.isArray(input.content)) {
    return input.content;
  }
  if ("parts" in input && Array.isArray(input.parts)) {
    return input.parts;
  }
  return [];
}

function isActiveStatus(status: string): boolean {
  return ["pending", "running", "waiting", "queued"].includes(status);
}

function DetailChip({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: React.ReactNode;
  tone?: "default" | "accent";
}) {
  const toneClassName =
    tone === "accent"
      ? "border-accent/30 bg-accent/10"
      : "border-line bg-panel-muted";

  return (
    <div className={`rounded-lg border px-3 py-2 ${toneClassName}`}>
      <div className="text-[11px] uppercase tracking-wide text-ink-faint">{label}</div>
      <div className="mt-1 break-all text-sm text-ink-soft">{value}</div>
    </div>
  );
}

function WakeConditionCard({ wc }: { wc: AgentStateDetail["wake_condition"] }) {
  if (!wc) return null;

  const waitsetProgress = getWaitsetProgress(wc);
  const timerLabel =
    wc.type === "timer" || wc.type === "periodic"
      ? formatWakeConditionTimer(wc)
      : null;

  return (
    <div className="space-y-2 rounded-lg border border-line bg-panel p-4">
      <h3 className="text-sm font-medium">Wake Condition</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
        <div>
          <span className="text-ink-faint">Type: </span>
          <span className="text-ink-soft">{wc.type}</span>
        </div>
        {wc.type === "waitset" && (
          <>
            <div>
              <span className="text-ink-faint">Progress: </span>
              <span className="text-ink-soft">
                {waitsetProgress.completed} / {waitsetProgress.total}
              </span>
            </div>
            <div>
              <span className="text-ink-faint">Mode: </span>
              <span className="text-ink-soft">{wc.wait_mode}</span>
            </div>
          </>
        )}
        {(wc.type === "timer" || wc.type === "periodic") && (
          <>
            {timerLabel && (
              <div>
                <span className="text-ink-faint">
                  {wc.type === "periodic" ? "Interval: " : "Duration: "}
                </span>
                <span className="text-ink-soft">{timerLabel}</span>
              </div>
            )}
            {wc.wakeup_at && (
              <div>
                <span className="text-ink-faint">Wakeup At: </span>
                <span className="text-ink-soft">
                  {formatLocalDateTime(wc.wakeup_at)}
                </span>
              </div>
            )}
          </>
        )}
        {wc.timeout_at && (
          <div>
            <span className="text-ink-faint">Timeout: </span>
            <span className="text-ink-soft">
              {formatLocalDateTime(wc.timeout_at)}
            </span>
          </div>
        )}
      </div>
      {wc.type === "waitset" && wc.wait_for.length > 0 && (
        <div className="mt-2">
          <div className="h-2 overflow-hidden rounded-full bg-panel-muted">
            <div
              className="h-full rounded-full bg-success transition-all"
              style={{ width: `${waitsetProgress.percent}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function ChildrenTable({ items }: { items: AgentStateListItem[] }) {
  if (items.length === 0) return null;
  return (
    <div className="overflow-hidden rounded-lg border border-line bg-panel">
      <div className="border-b border-line bg-panel-muted px-4 py-3">
        <h3 className="text-sm font-medium">
          Child Agents ({items.length})
        </h3>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-panel-muted text-xs uppercase tracking-wide text-ink-faint">
          <tr>
            <th className="text-left px-4 py-2">Agent</th>
            <th className="text-left px-4 py-2">Task</th>
            <th className="text-center px-4 py-2">Status</th>
            <th className="text-right px-4 py-2">Cost</th>
            <th className="text-right px-4 py-2">Tokens</th>
            <th className="text-left px-4 py-2">Result</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {items.map((c) => {
            const metrics = normalizeRunMetricsSummary(c.metrics);
            return (
              <tr key={c.id} className="transition-colors hover:bg-panel-muted">
                <td className="px-4 py-2.5">
                  <MonoLink href={`/scheduler/${c.id}`}>
                    {c.id}
                  </MonoLink>
                </td>
                <td className="px-4 py-2.5 max-w-xs">
                  <UserInputCompact input={c.task} maxLength={50} />
                </td>
                <td className="px-4 py-2.5 text-center">
                  <SchedulerStatusBadge status={c.status} />
                </td>
                <td className="px-4 py-2.5 text-right text-xs text-ink-soft">
                  {formatUsd(metrics.token_cost)}
                </td>
                <td className="px-4 py-2.5 text-right text-xs text-ink-faint">
                  {formatTokenCount(metrics.input_tokens)} / {formatTokenCount(metrics.output_tokens)} / {formatTokenCount(metrics.total_tokens)}
                </td>
                <td className="max-w-xs truncate px-4 py-2.5 text-xs text-ink-muted">
                  {c.result_summary || "-"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ControlPanel({
  state,
  onAction,
}: {
  state: AgentStateDetail;
  onAction: () => Promise<void>;
}) {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<{
    kind: "success" | "error" | "info";
    text: string;
  } | null>(null);
  const messageInputId = `scheduler-control-${state.id}`;

  const isRoot = state.parent_id === null;
  const isActive = isActiveStatus(state.status);
  const canResume = isRoot && state.is_persistent && ["idle", "failed"].includes(state.status);

  const handle = async (
    fn: () => Promise<unknown>,
    successText: string,
    clearInput = false,
  ) => {
    setBusy(true);
    try {
      await fn();
      if (clearInput) {
        setInput("");
      }
      setFeedback({ kind: "success", text: successText });
      await onAction();
    } catch (error) {
      const text =
        error instanceof Error ? error.message : "Operation failed";
      setFeedback({ kind: "error", text });
    } finally {
      setBusy(false);
    }
  };

  if (!isRoot) return null;

  return (
    <div className="rounded-lg border border-line bg-panel p-4 space-y-3">
      <h3 className="text-sm font-medium">Control</h3>
      <div className="flex gap-2">
        <label htmlFor={messageInputId} className="sr-only">
          {canResume ? "Resume message" : "Steering message"}
        </label>
        <input
          id={messageInputId}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            canResume
              ? "Resume message..."
              : isActive
                ? "Steer message..."
                : "This state is not accepting messages"
          }
          className="ui-input flex-1"
          onKeyDown={(e) => {
            if (e.key === "Enter" && input.trim()) {
              if (canResume) {
                void handle(
                  () => resumeAgent(state.id, input.trim()),
                  "Message submitted. Auto-refreshing until the agent settles.",
                  true,
                );
              } else if (isActive) {
                void handle(
                  () => steerAgent(state.id, input.trim(), true),
                  "Steer message queued. Auto-refreshing while the agent is active.",
                  true,
                );
              }
            }
          }}
        />
        {isActive && (
          <button
            type="button"
            disabled={busy || !input.trim()}
            onClick={() =>
              void handle(
                () => steerAgent(state.id, input.trim(), true),
                "Steer message queued. Auto-refreshing while the agent is active.",
                true,
              )
            }
            className="ui-button ui-button-primary min-h-10 px-3 py-1.5 text-xs"
          >
            {busy ? "Sending..." : "Steer"}
          </button>
        )}
        {canResume && (
          <button
            type="button"
            disabled={busy || !input.trim()}
            onClick={() =>
              void handle(
                () => resumeAgent(state.id, input.trim()),
                "Message submitted. Auto-refreshing until the agent settles.",
                true,
              )
            }
            className="ui-button ui-button-primary min-h-10 px-3 py-1.5 text-xs"
          >
            {busy ? "Sending..." : "Resume"}
          </button>
        )}
        {isActive && (
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              void handle(
                () => cancelAgent(state.id),
                "Cancel signal sent. Refreshing state.",
              )
            }
            className="ui-button ui-button-danger min-h-10 px-3 py-1.5 text-xs"
          >
            {busy ? "Working..." : "Cancel"}
          </button>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <PillBadge className="rounded bg-panel-strong px-2 py-1 text-ink-soft whitespace-nowrap">
          {canResume ? "next action: resume" : isActive ? "next action: steer" : "read only"}
        </PillBadge>
        <span className="text-ink-muted">
          {isActive
            ? "This page auto-refreshes while the agent is active."
            : "Idle and failed states accept a new resume message."}
        </span>
      </div>
      {feedback && (
        <div
          role={feedback.kind === "error" ? "alert" : "status"}
          className={`rounded border px-3 py-2 text-xs ${
            feedback.kind === "error"
              ? "border-danger/30 bg-danger/10 text-danger"
              : "border-success/30 bg-success/10 text-success"
          }`}
        >
          {feedback.text}
        </div>
      )}
    </div>
  );
}

export default function SchedulerDetailPage() {
  const params = useParams();
  const stateId = params.id as string;
  const isPageVisible = usePageVisibility();
  const [state, setState] = useState<AgentStateDetail | null>(null);
  const [agentConfig, setAgentConfig] = useState<AgentConfig | null>(null);
  const [children, setChildren] = useState<AgentStateListItem[]>([]);
  const [pendingEvents, setPendingEvents] = useState<PendingEventItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const loadStateData = useCallback(async (showLoading = true) => {
    if (showLoading) {
      setLoading(true);
    }

    try {
      const nextState = await getAgentState(stateId);
      const [nextChildren, nextPendingEvents] = await Promise.all([
        getAgentStateChildren(stateId),
        getPendingEvents(stateId),
      ]);
      const nextAgentConfig =
        nextState?.agent_config_id != null
          ? await getAgent(nextState.agent_config_id).catch(() => null)
          : null;
      setState(nextState);
      setAgentConfig(nextAgentConfig);
      setChildren(nextChildren);
      setPendingEvents(nextPendingEvents);
      setError(null);
    } catch (err) {
      setState(null);
      setChildren([]);
      setPendingEvents([]);
      setError(err instanceof Error ? err.message : "Failed to load agent state");
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, [stateId]);

  const refresh = useCallback(() => loadStateData(false), [loadStateData]);

  useEffect(() => {
    void loadStateData();
  }, [loadStateData]);

  useEffect(() => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (!state || !isActiveStatus(state.status) || !isPageVisible) {
      return;
    }
    pollTimerRef.current = window.setInterval(() => {
      void loadStateData(false);
    }, 2000);
    return () => {
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [isPageVisible, loadStateData, state]);

  useEffect(() => {
    if (isPageVisible && state && isActiveStatus(state.status)) {
      void loadStateData(false);
    }
  }, [isPageVisible, loadStateData, state]);

  if (loading) {
    return <FullPageMessage>Loading...</FullPageMessage>;
  }

  if (!state) {
    return <FullPageMessage>Agent state not found</FullPageMessage>;
  }

  const stateMetrics = normalizeRunMetricsSummary(state.metrics);
  const taskContext = extractTaskContext(state.task);
  const taskParts = extractTaskParts(state.task);
  const attachmentCount = taskParts.filter((part) => part.type !== "text").length;
  const taskContextEntries = Object.entries(taskContext.metadata);
  const hasDetailSignals =
    Boolean(taskContext.source) ||
    attachmentCount > 0 ||
    taskContextEntries.length > 0;

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {error && <ErrorStateMessage>{error}</ErrorStateMessage>}
      <BackHeader
        href="/scheduler"
        title="Agent State"
        subtitle={state.id}
        rightContent={
          <>
            <MonoLink
              href={`/scheduler/${state.root_state_id ?? state.id}/tree${
                state.root_state_id && state.root_state_id !== state.id
                  ? `?selected=${state.id}`
                  : ""
              }`}
            >
              Tree View
            </MonoLink>
            {state.is_persistent && (
              <PillBadge className="rounded bg-accent/15 px-1.5 py-0.5 text-xs whitespace-nowrap text-accent">
                persistent
              </PillBadge>
            )}
            <SchedulerStatusBadge status={state.status} />
          </>
        }
      />

      <TokenSummaryCards
        cost={stateMetrics.token_cost}
        inputTokens={stateMetrics.input_tokens}
        outputTokens={stateMetrics.output_tokens}
        totalTokens={stateMetrics.total_tokens}
        cacheReadTokens={stateMetrics.cache_read_tokens}
        cacheCreationTokens={stateMetrics.cache_creation_tokens}
        extraCards={
          <>
            <MetricCard
              label="Status"
              value={<SchedulerStatusBadge status={state.status} />}
            />
            <MetricCard
              label="Session"
              valueClassName="text-sm font-mono truncate"
              value={state.session_id}
            />
            <MetricCard
              label="Children"
              valueClassName="text-lg font-medium"
              value={children.length}
            />
            <MetricCard
              label="Pending Events"
              valueClassName="text-lg font-medium"
              value={pendingEvents.length}
            />
          </>
        }
      />

      <SectionCard
        title="Details"
        className="p-4"
        headerClassName="mb-2"
        titleClassName="text-sm font-medium"
      >
        <div className="space-y-4">
          <div className="rounded-lg border border-line bg-panel-muted p-3">
            <div className="mb-2 flex items-center justify-between gap-3">
              <div className="text-xs uppercase tracking-wide text-ink-faint">Task</div>
              {hasDetailSignals && (
                <div className="flex flex-wrap items-center gap-2">
                  {taskContext.source && (
                    <PillBadge className="rounded bg-panel-strong px-1.5 py-0.5 text-[10px] text-ink-soft whitespace-nowrap">
                      {taskContext.source}
                    </PillBadge>
                  )}
                  {attachmentCount > 0 && (
                    <PillBadge className="rounded bg-panel-strong px-1.5 py-0.5 text-[10px] text-ink-soft whitespace-nowrap">
                      attachments {attachmentCount}
                    </PillBadge>
                  )}
                </div>
              )}
            </div>
            <UserInputDetail input={state.task} showContext={false} maxTextLength={300} />
            {taskContextEntries.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
                {taskContextEntries.map(([key, value]) => (
                  <div key={key} className="text-xs text-ink-muted">
                    <span className="text-ink-faint">{key}:</span>{" "}
                    <span className="text-ink-soft">{String(value)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <DetailChip label="Model" value={agentConfig ? `${agentConfig.model_provider} / ${agentConfig.model_name}` : "Unavailable"} tone="accent" />
            <DetailChip
              label="Config"
              value={
                state.agent_config_id ? (
                  <MonoText>{state.agent_config_id}</MonoText>
                ) : (
                  "No linked config"
                )
              }
            />
            <DetailChip label="Depth / Wake" value={`${state.depth} / ${state.wake_count}`} />
            <DetailChip label="Signal" value={state.signal_propagated ? "propagated" : "not propagated"} />
            <DetailChip
              label="Session"
              value={<MonoText>{state.session_id}</MonoText>}
            />
            <DetailChip
              label="Parent"
              value={
                state.parent_id ? (
                  <MonoLink href={`/scheduler/${state.parent_id}`}>{state.parent_id}</MonoLink>
                ) : (
                  "Root agent"
                )
              }
            />
            <DetailChip label="Created" value={formatLocalDateTime(state.created_at)} />
            <DetailChip label="Updated" value={formatLocalDateTime(state.updated_at)} />
          </div>

          {agentConfig && (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <DetailChip label="Temperature" value={agentConfig.model_params?.temperature ?? "-"} />
              <DetailChip label="Max Output" value={agentConfig.model_params?.max_output_tokens ?? "-"} />
              <DetailChip label="Context Window" value={agentConfig.model_params?.max_context_window ?? "-"} />
              <DetailChip label="Tools" value={agentConfig.tools.length} />
              {agentConfig.model_params?.base_url && (
                <DetailChip label="Base URL" value={agentConfig.model_params.base_url} />
              )}
              {agentConfig.model_params?.api_key_env_name && (
                <DetailChip label="API Key Env" value={agentConfig.model_params.api_key_env_name} />
              )}
            </div>
          )}

          {state.pending_input && (
            <div className="rounded-lg border border-warning/30 bg-warning/10 p-3">
              <div className="mb-2 text-xs uppercase tracking-wide text-warning">
                Pending Input
              </div>
              <UserInputCompact
                input={state.pending_input}
                maxLength={240}
                showAttachmentBadge={true}
                showContextBadge={true}
                showMetadata={true}
              />
            </div>
          )}

          <InfoRow label="Run Metrics">
            <span className="text-ink-muted">
              runs {stateMetrics.run_count} | steps {stateMetrics.step_count} | tools {stateMetrics.tool_calls_count} | total {formatTokenCount(stateMetrics.total_tokens)} | duration {formatDurationMs(stateMetrics.duration_ms)}
            </span>
          </InfoRow>
        </div>
      </SectionCard>

      {state.result_summary && (
        <SectionCard
          title="Result Summary"
          className="p-4"
          headerClassName="mb-2"
          titleClassName="text-sm font-medium"
        >
          <p className="text-sm whitespace-pre-wrap text-ink-soft">
            {state.result_summary}
          </p>
        </SectionCard>
      )}

      <ControlPanel state={state} onAction={refresh} />

      <WakeConditionCard wc={state.wake_condition} />

      {pendingEvents.length > 0 && (
        <SectionCard
          title={`Pending Events (${pendingEvents.length})`}
          className="p-4"
          headerClassName="mb-2"
          titleClassName="text-sm font-medium"
        >
          <div className="space-y-3">
            {pendingEvents.map((event) => (
              <div
                key={event.id}
                className="rounded-lg border border-line bg-panel-muted p-3"
              >
                <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
                  <PillBadge className="rounded bg-panel-strong px-1.5 py-0.5 text-[10px] text-ink-soft whitespace-nowrap">
                    {event.event_type}
                  </PillBadge>
                  <span className="font-mono text-ink-faint">{event.id}</span>
                  {event.source_agent_id && (
                    <span className="text-ink-faint">
                      from {event.source_agent_id}
                    </span>
                  )}
                  <span className="ml-auto text-ink-faint">
                    {formatLocalDateTime(event.created_at)}
                  </span>
                </div>
                <JsonDisclosure className="mt-2" label="Event payload" value={event.payload} />
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {Object.keys(state.config_overrides).length > 0 && (
        <SectionCard
          title="Config Overrides"
          className="p-4"
          headerClassName="mb-2"
          titleClassName="text-sm font-medium"
        >
          <JsonDisclosure label="Overrides JSON" value={state.config_overrides} />
        </SectionCard>
      )}

      <ChildrenTable items={children} />
    </div>
  );
}
