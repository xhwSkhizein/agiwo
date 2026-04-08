"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Loader2,
  Moon,
  PanelRight,
  Play,
  Workflow,
  XCircle,
} from "lucide-react";
import {
  cancelSession,
  createAgentSession,
  getAgent,
  getAgentStateChildren,
  getPendingEvents,
  getSessionDetail,
  getSessionSteps,
  sessionInputStreamUrl,
} from "@/lib/api";
import { ChatInputBar } from "@/components/chat-input-bar";
import { ChatMessageItem } from "@/components/chat-message";
import { PillBadge } from "@/components/pill-badge";
import { EmptyStateMessage, ErrorStateMessage, FullPageMessage } from "@/components/state-message";
import { UserInputCompact } from "@/components/user-input-detail";
import { SessionPanel } from "@/components/session-panel/session-panel";
import { useChatStream } from "@/hooks/use-chat-stream";
import type {
  AgentConfig,
  AgentStateDetail,
  AgentStateListItem,
  PendingEventItem,
  RunCompletedEventPayload,
  StreamEventPayload,
} from "@/lib/api";
import type { ChatMessage } from "@/lib/chat-types";
import {
  appendUnseenStepMessages,
  contentToText,
  genMessageId,
  getHighestMessageSequence,
  messagesFromSteps,
} from "@/lib/chat-types";
import { formatWakeConditionSummary } from "@/lib/wake-condition";

type OrchestrationStatus =
  | "idle"
  | "running"
  | "waiting"
  | "queued"
  | "completed"
  | "failed"
  | "cancelled";

const statusConfig: Record<
  OrchestrationStatus,
  { icon: typeof Play; color: string; label: string }
> = {
  idle: { icon: Clock, color: "text-zinc-500", label: "Idle" },
  running: { icon: Loader2, color: "text-blue-400", label: "Running" },
  waiting: { icon: Moon, color: "text-yellow-400", label: "Waiting" },
  queued: { icon: Clock, color: "text-cyan-400", label: "Queued" },
  completed: {
    icon: CheckCircle2,
    color: "text-green-400",
    label: "Completed",
  },
  failed: { icon: AlertCircle, color: "text-red-400", label: "Failed" },
  cancelled: { icon: XCircle, color: "text-zinc-400", label: "Cancelled" },
};

const childStatusBadge: Record<string, string> = {
  pending: "bg-zinc-700 text-zinc-300",
  running: "bg-blue-900/50 text-blue-400",
  waiting: "bg-yellow-900/50 text-yellow-400",
  idle: "bg-purple-900/50 text-purple-400",
  queued: "bg-cyan-900/50 text-cyan-300",
  completed: "bg-green-900/50 text-green-400",
  failed: "bg-red-900/50 text-red-400",
};

function statusFromSchedulerState(
  state: AgentStateDetail | null,
  fallback: OrchestrationStatus = "idle",
): OrchestrationStatus {
  if (!state) {
    return fallback;
  }
  switch (state.status) {
    case "running":
      return "running";
    case "waiting":
      return "waiting";
    case "queued":
      return "queued";
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    case "idle":
      return "idle";
    default:
      return fallback;
  }
}

/**
 * Render the scheduler chat page UI for interacting with an agent's sessions, orchestration state, and child agents.
 *
 * Provides a full client-side interface that:
 * - loads agent metadata and session state,
 * - manages session selection, creation, and URL synchronization,
 * - streams and paginates session messages while reconciling remote session steps,
 * - polls and displays orchestration/root state, pending events, and child agents (with per-child streamed messages),
 * - exposes controls for sending messages, cancelling runs, and toggling session/child panels.
 *
 * @returns The rendered scheduler chat page UI element.
 */
function SchedulerChatPageContent() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const agentId = params.id as string;

  const [agent, setAgent] = useState<AgentConfig | null>(null);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(
    searchParams.get("session"),
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [orchestrationStatus, setOrchestrationStatus] =
    useState<OrchestrationStatus>("idle");
  const [stateId, setStateId] = useState<string | null>(null);
  const [rootState, setRootState] = useState<AgentStateDetail | null>(null);
  const [pendingEvents, setPendingEvents] = useState<PendingEventItem[]>([]);
  const [children, setChildren] = useState<AgentStateListItem[]>([]);
  const [expandedChildren, setExpandedChildren] = useState<Set<string>>(
    new Set(),
  );
  const [childMessages, setChildMessages] = useState<
    Record<string, ChatMessage[]>
  >({});
  const [showSessionPanel, setShowSessionPanel] = useState(
    () => searchParams.get("session") === null,
  );
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  const isStreamingRef = useRef(false);
  const loadHistoryMessagesRef = useRef<(history: ChatMessage[]) => void>(() => {});
  const handleSessionCreatedRef = useRef<(sessionId: string) => void>(() => {});
  const creatingSessionRef = useRef(false);

  const updateSessionUrl = useCallback(
    (sid: string) => {
      const url = new URL(window.location.href);
      url.searchParams.set("session", sid);
      router.replace(url.pathname + url.search);
    },
    [router],
  );

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const syncOrchestrationState = useCallback(
    async (targetSessionId: string) => {
      const detail = await getSessionDetail(targetSessionId);
      setRootState(detail.scheduler_state);
      setStateId(detail.scheduler_state?.id ?? null);
      setOrchestrationStatus(statusFromSchedulerState(detail.scheduler_state));
      if (!detail.scheduler_state?.id) {
        setPendingEvents([]);
        setChildren([]);
        stopPolling();
        return;
      }
      const [nextChildren, nextPendingEvents] = await Promise.all([
        getAgentStateChildren(detail.scheduler_state.id),
        getPendingEvents(detail.scheduler_state.id),
      ]);
      setChildren(nextChildren);
      setPendingEvents(nextPendingEvents);
    },
    [stopPolling],
  );

  const reconcileMessages = useCallback(
    async (targetSessionId: string) => {
      if (isStreamingRef.current) {
        return;
      }
      const highestSequence = getHighestMessageSequence(messagesRef.current);
      if (highestSequence === null) {
        const latest = await getSessionSteps(targetSessionId, {
          limit: 100,
          order: "desc",
        });
        loadHistoryMessagesRef.current(messagesFromSteps([...latest.items].reverse()));
        setHistoryHasMore(latest.has_more);
        return;
      }

      const newer = await getSessionSteps(targetSessionId, {
        start_seq: highestSequence + 1,
        limit: 1000,
        order: "asc",
      });
      if (newer.items.length === 0) {
        return;
      }
      loadHistoryMessagesRef.current(
        appendUnseenStepMessages(messagesRef.current, newer.items),
      );
    },
    [],
  );

  const pollOrchestration = useCallback(
    (nextStateId: string, targetSessionId: string) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const [detail, nextChildren, nextPendingEvents] = await Promise.all([
            getSessionDetail(targetSessionId),
            getAgentStateChildren(nextStateId),
            getPendingEvents(nextStateId),
          ]);
          setRootState(detail.scheduler_state);
          setChildren(nextChildren);
          setPendingEvents(nextPendingEvents);
          setOrchestrationStatus(statusFromSchedulerState(detail.scheduler_state));
          if (!isStreamingRef.current) {
            await reconcileMessages(targetSessionId);
          }
          if (!detail.scheduler_state || ["completed", "failed", "idle"].includes(detail.scheduler_state.status)) {
            stopPolling();
          }
        } catch {}
      }, 2000);
    },
    [reconcileMessages, stopPolling],
  );

  const handleChildEvent = useCallback(
    (childAgentId: string, data: StreamEventPayload) => {
      if (data.type === "step_delta" && "delta" in data && data.delta) {
        const delta = data.delta;
        if (delta.content) {
          setChildMessages((prev) => {
            const messages = prev[childAgentId] || [];
            const last = messages[messages.length - 1];
            if (last && last.role === "assistant" && last.isStreaming) {
              return {
                ...prev,
                [childAgentId]: messages.map((message, index) =>
                  index === messages.length - 1
                    ? { ...message, text: `${message.text || ""}${delta.content}` }
                    : message,
                ),
              };
            }
            return {
              ...prev,
              [childAgentId]: [
                ...messages,
                {
                  id: genMessageId(),
                  role: "assistant",
                  text: delta.content ?? "",
                  isStreaming: true,
                  sourceAgentId: childAgentId,
                },
              ],
            };
          });
        }
      }

      if (data.type === "step_completed" && "step" in data && data.step) {
        const step = data.step;
        if (step.role === "tool") {
          setChildMessages((prev) => ({
            ...prev,
            [childAgentId]: [
              ...(prev[childAgentId] || []),
              {
                id: genMessageId(),
                role: "tool",
                text: contentToText(step.content) ?? "",
                rawContent: step.content,
                name: step.name || undefined,
                sourceAgentId: childAgentId,
              },
            ],
          }));
        }
      }

      if (data.type === "run_completed") {
        setChildMessages((prev) => {
          const messages = prev[childAgentId] || [];
          return {
            ...prev,
            [childAgentId]: messages.map((message) =>
              message.isStreaming ? { ...message, isStreaming: false } : message,
            ),
          };
        });
      }
    },
    [],
  );

  const {
    messages,
    isStreaming,
    sendMessage,
    clearMessages,
    loadHistoryMessages,
  } = useChatStream(sessionId ? sessionInputStreamUrl(sessionId) : "", {
    onSessionCaptured: (sid) => {
      if (!sessionId) {
        setSessionId(sid);
        updateSessionUrl(sid);
      }
    },
    onRootStateCaptured: (capturedStateId) => {
      setStateId(capturedStateId);
      if (sessionId) {
        pollOrchestration(capturedStateId, sessionId);
      }
    },
    onChildEvent: handleChildEvent,
    onSchedulerFailed: (message) => {
      setError(message);
      setOrchestrationStatus("failed");
      stopPolling();
    },
    onRunStarted: () => {
      setOrchestrationStatus("running");
      setStartTime(Date.now());
    },
    onRunCompleted: async (event: RunCompletedEventPayload) => {
      if (event.depth !== 0 || !sessionId) {
        return;
      }
      if (event.termination_reason === "sleeping") {
        setOrchestrationStatus("waiting");
      }
      await syncOrchestrationState(sessionId);
      if (!isStreamingRef.current) {
        await reconcileMessages(sessionId);
      }
    },
  });

  useEffect(() => {
    return stopPolling;
  }, [stopPolling]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    loadHistoryMessagesRef.current = loadHistoryMessages;
  }, [loadHistoryMessages]);

  useEffect(() => {
    getAgent(agentId)
      .then(setAgent)
      .catch(() => setAgent(null))
      .finally(() => setLoading(false));
  }, [agentId]);

  useEffect(() => {
    if (!sessionId) {
      clearMessages();
      setRootState(null);
      setStateId(null);
      setPendingEvents([]);
      setChildren([]);
      setHistoryHasMore(false);
      setOrchestrationStatus("idle");
      stopPolling();
      return;
    }
    if (messages.length > 0 || isStreaming) {
      return;
    }
    Promise.all([
      syncOrchestrationState(sessionId),
      getSessionSteps(sessionId, { limit: 100, order: "desc" }),
    ])
      .then(([, stepsPage]) => {
        loadHistoryMessages(messagesFromSteps([...stepsPage.items].reverse()));
        setHistoryHasMore(stepsPage.has_more);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to hydrate session");
      });
  }, [
    clearMessages,
    isStreaming,
    loadHistoryMessages,
    messages.length,
    sessionId,
    stopPolling,
    syncOrchestrationState,
  ]);

  useEffect(() => {
    if (stateId && sessionId && ["running", "waiting", "queued"].includes(orchestrationStatus)) {
      pollOrchestration(stateId, sessionId);
      return;
    }
    if (!["running", "waiting", "queued"].includes(orchestrationStatus)) {
      stopPolling();
    }
  }, [orchestrationStatus, pollOrchestration, sessionId, stateId, stopPolling]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!startTime || ["completed", "failed", "cancelled", "idle"].includes(orchestrationStatus)) {
      return;
    }
    const timer = setInterval(() => {
      setElapsed((Date.now() - startTime) / 1000);
    }, 100);
    return () => clearInterval(timer);
  }, [orchestrationStatus, startTime]);

  const handleSend = () => {
    const text = input.trim();
    if (!sessionId) {
      setError("Select an existing session or create a new session first.");
      return;
    }
    if (!text || isStreaming) return;
    setInput("");
    setError(null);
    setOrchestrationStatus("running");
    setStartTime(Date.now());
    setChildren([]);
    setPendingEvents([]);
    setChildMessages({});
    sendMessage(text);
  };

  const handleCancel = async () => {
    if (!sessionId) return;
    try {
      await cancelSession(sessionId);
      setOrchestrationStatus("cancelled");
      stopPolling();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cancel");
    }
  };

  const toggleChild = (childId: string) => {
    setExpandedChildren((prev) => {
      const next = new Set(prev);
      if (next.has(childId)) next.delete(childId);
      else next.add(childId);
      return next;
    });
  };

  const handleSessionCreated = useCallback(
    (newSessionId: string) => {
      setSessionId(newSessionId);
      updateSessionUrl(newSessionId);
      setRefreshTrigger((t) => t + 1);
      clearMessages();
      setError(null);
      setHistoryHasMore(false);
      setChildren([]);
      setPendingEvents([]);
      setChildMessages({});
      setRootState(null);
      setStateId(null);
      setOrchestrationStatus("idle");
    },
    [updateSessionUrl, clearMessages],
  );

  // Keep ref in sync for auto-create effect (avoids circular deps)
  useEffect(() => {
    handleSessionCreatedRef.current = handleSessionCreated;
  }, [handleSessionCreated]);

  const handleSessionForked = useCallback(
    (forkedSessionId: string) => {
      handleSessionCreated(forkedSessionId);
    },
    [handleSessionCreated],
  );

  // Auto-create session if none exists
  useEffect(() => {
    if (!agent || sessionId || creatingSessionRef.current) return;

    creatingSessionRef.current = true;
    let cancelled = false;
    const autoCreate = async () => {
      try {
        const result = await createAgentSession(agentId);
        if (!cancelled) {
          handleSessionCreatedRef.current(result.session_id);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to create session");
        }
      } finally {
        creatingSessionRef.current = false;
      }
    };

    void autoCreate();
    return () => {
      cancelled = true;
    };
    // Note: handleSessionCreatedRef is stable; handleSessionCreated intentionally excluded
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent, agentId, sessionId]);

  const handleSessionSwitch = async (targetSessionId: string) => {
    setSessionId(targetSessionId);
    updateSessionUrl(targetSessionId);
    clearMessages();
    setChildMessages({});
    setExpandedChildren(new Set());
    try {
      const [detail, stepsPage] = await Promise.all([
        syncOrchestrationState(targetSessionId),
        getSessionSteps(targetSessionId, { limit: 100, order: "desc" }),
      ]);
      void detail;
      loadHistoryMessages(messagesFromSteps([...stepsPage.items].reverse()));
      setHistoryHasMore(stepsPage.has_more);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to switch session");
    }
  };

  const loadEarlierMessages = async () => {
    if (!sessionId || messages.length === 0 || loadingHistory) {
      return;
    }
    const firstSequence = messages[0]?.sequence ?? null;
    if (!firstSequence || firstSequence <= 1) {
      setHistoryHasMore(false);
      return;
    }
    setLoadingHistory(true);
    try {
      const older = await getSessionSteps(sessionId, {
        limit: 100,
        order: "desc",
        end_seq: firstSequence - 1,
      });
      loadHistoryMessages([
        ...messagesFromSteps([...older.items].reverse()),
        ...messages,
      ]);
      setHistoryHasMore(older.has_more);
    } finally {
      setLoadingHistory(false);
    }
  };

  const StatusIcon = statusConfig[orchestrationStatus].icon;
  const childCount = useMemo(() => children.length, [children.length]);

  if (loading) {
    return <FullPageMessage>Loading agent...</FullPageMessage>;
  }

  if (!agent) {
    return <FullPageMessage>Agent not found</FullPageMessage>;
  }

  return (
    <div className="flex h-full">
      <div className="flex flex-col flex-1 min-w-0">
        <div className="shrink-0 px-5 py-3 border-b border-zinc-800 flex items-center justify-between bg-zinc-900/50">
          <div className="flex items-center gap-3">
            <Link
              href="/agents"
              aria-label="Back to agents"
              className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
            </Link>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-medium">{agent.name}</h1>
                <PillBadge className="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-300 font-medium whitespace-nowrap">
                  Scheduler
                </PillBadge>
              </div>
              <p className="text-xs text-zinc-500">
                {agent.model_provider}/{agent.model_name}
                {sessionId && (
                  <span className="ml-2 font-mono">{sessionId.slice(0, 8)}</span>
                )}
                {!sessionId && <span className="ml-2">No session selected</span>}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {sessionId && stateId && ["running", "waiting", "queued"].includes(orchestrationStatus) && (
              <button
                onClick={handleCancel}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-900/30 text-red-400 text-xs font-medium hover:bg-red-900/50 transition-colors"
              >
                <XCircle className="w-3.5 h-3.5" />
                Cancel
              </button>
            )}
            <button
              type="button"
              onClick={() => setShowSessionPanel((v) => !v)}
              aria-label={showSessionPanel ? "Hide sessions" : "Show sessions"}
              className={`p-1.5 rounded transition-colors ${
                showSessionPanel
                  ? "bg-zinc-700 text-white"
                  : "hover:bg-zinc-800 text-zinc-500"
              }`}
            >
              <PanelRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto px-5 py-4 space-y-4">
          {error && <ErrorStateMessage>{error}</ErrorStateMessage>}

          {historyHasMore && (
            <button
              type="button"
              onClick={loadEarlierMessages}
              disabled={loadingHistory}
              className="rounded-md border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 hover:border-zinc-700 hover:text-zinc-200 disabled:opacity-40"
            >
              {loadingHistory ? "Loading..." : "Load earlier"}
            </button>
          )}

          {!sessionId && (
            <EmptyStateMessage className="flex h-full flex-col items-center justify-center gap-3 text-sm text-zinc-500">
              Select an existing session or create a new one to start the scheduler conversation.
              <button
                type="button"
                onClick={() => setShowSessionPanel(true)}
                className="rounded-md border border-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:border-zinc-700 hover:text-white"
              >
                Open Sessions
              </button>
            </EmptyStateMessage>
          )}

          {sessionId && messages.length === 0 && (
            <EmptyStateMessage className="flex items-center justify-center h-full text-zinc-600 text-sm">
              Send a message to start the scheduler orchestration
            </EmptyStateMessage>
          )}

          {messages.map((msg) => (
            <ChatMessageItem key={msg.id} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="shrink-0 px-5 py-3 border-t border-zinc-800 bg-zinc-900/50">
          <ChatInputBar
            value={input}
            onChange={setInput}
            onSubmit={handleSend}
            disabled={isStreaming || !sessionId}
            label="Scheduler message"
            submitLabel="Send scheduler message"
          />
        </div>
      </div>

      {showSessionPanel && (
        <div className="w-80 shrink-0 border-l border-zinc-800 bg-zinc-950/50">
          <SessionPanel
            agentId={agentId}
            currentSessionId={sessionId}
            onSwitch={handleSessionSwitch}
            onCreated={handleSessionCreated}
            onForked={handleSessionForked}
            refreshTrigger={refreshTrigger}
            onDelete={(deletedSessionId) => {
              if (deletedSessionId === sessionId) {
                // 如果删除的是当前 session，清空状态
                setSessionId(null);
                const url = new URL(window.location.href);
                url.searchParams.delete("session");
                router.replace(url.pathname + url.search);
                clearMessages();
                setRootState(null);
                setStateId(null);
                setOrchestrationStatus("idle");
              }
            }}
          />
        </div>
      )}

      <div className="w-96 shrink-0 border-l border-zinc-800 flex flex-col overflow-hidden bg-zinc-950/50">
        <div className="px-4 py-3 border-b border-zinc-800 space-y-2">
          <div className="flex items-center gap-2">
            <StatusIcon
              className={`w-4 h-4 ${statusConfig[orchestrationStatus].color} ${
                orchestrationStatus === "running" ? "animate-spin" : ""
              }`}
            />
            <span
              className={`text-sm font-medium ${statusConfig[orchestrationStatus].color}`}
            >
              {statusConfig[orchestrationStatus].label}
            </span>
            {startTime && (
              <span className="text-xs text-zinc-500 ml-auto font-mono">
                {elapsed.toFixed(1)}s
              </span>
            )}
          </div>

          {stateId && (
            <div className="space-y-1">
              <p className="text-[10px] text-zinc-600 font-mono truncate">
                State: {stateId}
              </p>
              <div className="flex flex-wrap gap-2 text-[11px] text-zinc-500">
                <Link
                  href={`/scheduler/${stateId}`}
                  className="rounded border border-zinc-800 px-2 py-1 hover:border-zinc-700 hover:text-zinc-300"
                >
                  Open state
                </Link>
                {sessionId && (
                  <Link
                    href={`/sessions/${sessionId}`}
                    className="rounded border border-zinc-800 px-2 py-1 hover:border-zinc-700 hover:text-zinc-300"
                  >
                    Open session
                  </Link>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="flex-1 overflow-auto space-y-4 p-4">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 space-y-2">
            <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-zinc-500">
              <Workflow className="w-3.5 h-3.5" />
              Root State
            </div>
            {rootState ? (
              <>
                <UserInputCompact input={rootState.task} maxLength={140} />
                {rootState.wake_condition && (
                  <p className="text-xs text-zinc-500">
                    {formatWakeConditionSummary(rootState.wake_condition)}
                  </p>
                )}
                {rootState.result_summary && (
                  <p className="text-xs text-zinc-400 whitespace-pre-wrap">
                    {rootState.result_summary}
                  </p>
                )}
              </>
            ) : (
              <p className="text-xs text-zinc-600">No active root state</p>
            )}
          </div>

          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-xs font-medium text-zinc-400 uppercase">
                Pending Events
              </h2>
              <span className="text-[11px] text-zinc-500">{pendingEvents.length}</span>
            </div>
            {pendingEvents.length === 0 ? (
              <p className="text-xs text-zinc-600">No pending events</p>
            ) : (
              <div className="space-y-2">
                {pendingEvents.map((event) => (
                  <div key={event.id} className="rounded border border-zinc-800 px-2 py-2 text-xs">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-zinc-300">{event.event_type}</span>
                      <span className="text-zinc-600 font-mono">{event.id.slice(0, 8)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-2">
            <h2 className="text-xs font-medium text-zinc-400 uppercase">
              Child Agents
              <span className="ml-1 text-zinc-500">({childCount})</span>
            </h2>
            {children.length === 0 ? (
              <p className="text-xs text-zinc-600">
                {orchestrationStatus === "idle"
                  ? "No orchestration running"
                  : "No child agents spawned yet"}
              </p>
            ) : (
              children.map((child) => (
                <div
                  key={child.id}
                  className="rounded-lg bg-zinc-900/50 border border-zinc-800"
                >
                  <button
                    type="button"
                    aria-expanded={expandedChildren.has(child.id)}
                    onClick={() => toggleChild(child.id)}
                    className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-zinc-800/50 transition-colors rounded-lg"
                  >
                    {expandedChildren.has(child.id) ? (
                      <ChevronDown className="w-3 h-3 text-zinc-500 shrink-0" />
                    ) : (
                      <ChevronRight className="w-3 h-3 text-zinc-500 shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-zinc-300 truncate">
                          {child.id.length > 20 ? `...${child.id.slice(-16)}` : child.id}
                        </span>
                        <PillBadge
                          className={`text-[10px] px-1.5 py-0.5 rounded font-medium shrink-0 whitespace-nowrap ${
                            childStatusBadge[child.status] || "bg-zinc-700 text-zinc-300"
                          }`}
                        >
                          {child.status}
                        </PillBadge>
                      </div>
                      <UserInputCompact input={child.task} maxLength={60} />
                    </div>
                  </button>

                  {expandedChildren.has(child.id) && (
                    <div className="px-3 pb-3 border-t border-zinc-800">
                      {child.result_summary && (
                        <div className="mt-2 text-xs text-zinc-400 whitespace-pre-wrap max-h-32 overflow-auto">
                          <span className="text-zinc-500 font-medium">Result: </span>
                          {child.result_summary.slice(0, 500)}
                        </div>
                      )}
                      {(childMessages[child.id] || []).length > 0 && (
                        <div className="mt-2 space-y-2 max-h-48 overflow-auto">
                          {(childMessages[child.id] || []).map((message) => (
                            <div key={message.id} className="text-[11px]">
                              <span
                                className={`font-medium ${
                                  message.role === "assistant"
                                    ? "text-green-400"
                                    : message.role === "tool"
                                      ? "text-amber-400"
                                      : "text-zinc-400"
                                }`}
                              >
                                {message.role}
                                {message.name ? ` — ${message.name}` : ""}:
                              </span>{" "}
                              <span className="text-zinc-400 whitespace-pre-wrap">
                                {(message.text || "").slice(0, 300)}
                                {(message.text || "").length > 300 ? "..." : ""}
                              </span>
                              {message.isStreaming && (
                                <Loader2 className="inline w-2.5 h-2.5 ml-1 animate-spin text-zinc-500" />
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function SchedulerChatPage() {
  return (
    <Suspense fallback={<FullPageMessage>Loading agent...</FullPageMessage>}>
      <SchedulerChatPageContent />
    </Suspense>
  );
}
