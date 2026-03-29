"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Loader2,
  XCircle,
  Clock,
  CheckCircle2,
  AlertCircle,
  Moon,
  Play,
  ChevronDown,
  ChevronRight,
  PanelRight,
} from "lucide-react";
import {
  getAgent,
  schedulerChatStreamUrl,
  cancelSchedulerChat,
  getAgentStateChildren,
  getSessionSteps,
} from "@/lib/api";
import { ChatInputBar } from "@/components/chat-input-bar";
import { ChatMessageItem } from "@/components/chat-message";
import { PillBadge } from "@/components/pill-badge";
import { EmptyStateMessage, FullPageMessage } from "@/components/state-message";
import { UserInputCompact } from "@/components/user-input-detail";
import { SessionPanel } from "@/components/session-panel/session-panel";
import { useChatStream } from "@/hooks/use-chat-stream";
import type { ChatMessage } from "@/lib/chat-types";
import { genMessageId } from "@/lib/chat-types";
import type {
  AgentConfig,
  AgentStateListItem,
  RunCompletedEventPayload,
  StreamEventPayload,
  StepResponse,
} from "@/lib/api";

type OrchestrationStatus =
  | "idle"
  | "running"
  | "waiting"
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

function stepsToMessages(steps: StepResponse[]): ChatMessage[] {
  const msgs: ChatMessage[] = [];
  for (const step of steps) {
    if (step.role === "user" && step.user_input) {
      const text =
        typeof step.user_input === "string"
          ? step.user_input
          : JSON.stringify(step.user_input);
      if (text) msgs.push({ id: genMessageId(), role: "user", content: text });
    } else if (step.role === "assistant") {
      const content =
        typeof step.content === "string"
          ? step.content
          : step.content
            ? JSON.stringify(step.content)
            : "";
      if (content || (step.tool_calls && step.tool_calls.length > 0)) {
        msgs.push({
          id: genMessageId(),
          role: "assistant",
          content: content || "",
          tool_calls: step.tool_calls ?? undefined,
          reasoning_content: step.reasoning_content ?? undefined,
        });
      }
    } else if (step.role === "tool") {
      const content =
        typeof step.content === "string"
          ? step.content
          : JSON.stringify(step.content);
      msgs.push({
        id: genMessageId(),
        role: "tool",
        content,
        name: step.name || undefined,
      });
    }
  }
  return msgs;
}

export default function SchedulerChatPage() {
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
  const [orchestrationStatus, setOrchestrationStatus] =
    useState<OrchestrationStatus>("idle");
  const [stateId, setStateId] = useState<string | null>(null);
  const [children, setChildren] = useState<AgentStateListItem[]>([]);
  const [expandedChildren, setExpandedChildren] = useState<Set<string>>(
    new Set(),
  );
  const [childMessages, setChildMessages] = useState<
    Record<string, ChatMessage[]>
  >({});
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [showSessionPanel, setShowSessionPanel] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const updateSessionUrl = useCallback(
    (sid: string) => {
      const url = new URL(window.location.href);
      url.searchParams.set("session", sid);
      router.replace(url.pathname + url.search);
    },
    [router],
  );

  const pollChildren = useCallback((sid: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const result = await getAgentStateChildren(sid);
        setChildren(result);
      } catch {}
    }, 2000);
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    return stopPolling;
  }, [stopPolling]);

  const handleChildEvent = useCallback(
    (childAgentId: string, data: StreamEventPayload) => {
      const type = data.type;
      if (type === "step_delta" && "delta" in data && data.delta) {
        const delta = data.delta;
        if (delta.content) {
          setChildMessages((prev) => {
            const msgs = prev[childAgentId] || [];
            const last = msgs[msgs.length - 1];
            if (last && last.role === "assistant" && last.isStreaming) {
              return {
                ...prev,
                [childAgentId]: msgs.map((m, i) =>
                  i === msgs.length - 1
                    ? { ...m, content: m.content + delta.content }
                    : m,
                ),
              };
            }
            return {
              ...prev,
              [childAgentId]: [
                ...msgs,
                {
                  id: genMessageId(),
                  role: "assistant" as const,
                  content: delta.content ?? "",
                  isStreaming: true,
                  agentId: childAgentId,
                },
              ],
            };
          });
        }
      }
      if (type === "step_completed" && "step" in data && data.step) {
        const step = data.step as StepResponse;
        if (step.role === "tool") {
          setChildMessages((prev) => ({
            ...prev,
            [childAgentId]: [
              ...(prev[childAgentId] || []),
              {
                id: genMessageId(),
                role: "tool" as const,
                content:
                  typeof step.content === "string"
                    ? step.content
                    : JSON.stringify(step.content),
                name: step.name || undefined,
                agentId: childAgentId,
              },
            ],
          }));
        }
        if (step.role === "assistant" && step.tool_calls?.length) {
          for (const call of step.tool_calls) {
            if (call.function?.name === "sleep_and_wait") {
              setOrchestrationStatus("waiting");
            }
          }
        }
      }
      if (type === "run_completed") {
        setChildMessages((prev) => {
          const msgs = prev[childAgentId] || [];
          return {
            ...prev,
            [childAgentId]: msgs.map((m) =>
              m.isStreaming ? { ...m, isStreaming: false } : m,
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
  } = useChatStream(schedulerChatStreamUrl(agentId), {
    onSessionCaptured: (sid) => {
      if (!sessionId) {
        setSessionId(sid);
        updateSessionUrl(sid);
      }
    },
    onRootAgentCaptured: (aid) => {
      setStateId(aid);
      pollChildren(aid);
    },
    onChildEvent: handleChildEvent,
    onSchedulerFailed: (error) => {
      setOrchestrationStatus("failed");
      stopPolling();
    },
    onRunStarted: () => {
      setOrchestrationStatus("running");
    },
    onRunCompleted: (event: RunCompletedEventPayload) => {
      if (event.depth === 0) {
        if (event.termination_reason === "sleeping") {
          setOrchestrationStatus("waiting");
        } else {
          setOrchestrationStatus("completed");
          stopPolling();
          if (stateId) {
            getAgentStateChildren(stateId)
              .then(setChildren)
              .catch(() => {});
          }
        }
      }
    },
  });

  useEffect(() => {
    getAgent(agentId)
      .then(setAgent)
      .catch(() => setAgent(null))
      .finally(() => setLoading(false));
  }, [agentId]);

  useEffect(() => {
    if (sessionId && messages.length === 0 && !isStreaming) {
      getSessionSteps(sessionId)
        .then((steps) => {
          const history = stepsToMessages(steps);
          if (history.length > 0) loadHistoryMessages(history);
        })
        .catch(() => {});
    }
  }, [sessionId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (
      !startTime ||
      orchestrationStatus === "completed" ||
      orchestrationStatus === "failed"
    )
      return;
    const timer = setInterval(() => {
      setElapsed((Date.now() - startTime) / 1000);
    }, 100);
    return () => clearInterval(timer);
  }, [startTime, orchestrationStatus]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");
    setOrchestrationStatus("running");
    setStartTime(Date.now());
    setChildren([]);
    setChildMessages({});
    sendMessage(text, sessionId);
  };

  const handleCancel = async () => {
    if (!stateId) return;
    try {
      await cancelSchedulerChat(agentId, stateId);
      setOrchestrationStatus("cancelled");
      stopPolling();
    } catch {}
  };

  const toggleChild = (childId: string) => {
    setExpandedChildren((prev) => {
      const next = new Set(prev);
      if (next.has(childId)) next.delete(childId);
      else next.add(childId);
      return next;
    });
  };

  const handleSessionSwitch = async (targetSessionId: string) => {
    setSessionId(targetSessionId);
    updateSessionUrl(targetSessionId);
    clearMessages();
    setOrchestrationStatus("idle");
    setChildren([]);
    setChildMessages({});
    setStateId(null);
    try {
      const steps = await getSessionSteps(targetSessionId);
      loadHistoryMessages(stepsToMessages(steps));
    } catch {}
  };

  const handleSessionCreated = (newSessionId: string) => {
    setSessionId(newSessionId);
    updateSessionUrl(newSessionId);
    clearMessages();
    setOrchestrationStatus("idle");
    setChildren([]);
    setChildMessages({});
    setStateId(null);
  };

  const handleSessionForked = (forkedSessionId: string) => {
    setSessionId(forkedSessionId);
    updateSessionUrl(forkedSessionId);
    clearMessages();
    setOrchestrationStatus("idle");
    setChildren([]);
    setChildMessages({});
    setStateId(null);
  };

  if (loading) {
    return <FullPageMessage>Loading agent...</FullPageMessage>;
  }

  if (!agent) {
    return <FullPageMessage>Agent not found</FullPageMessage>;
  }

  const StatusIcon = statusConfig[orchestrationStatus].icon;

  return (
    <div className="flex h-full">
      {/* Left: Chat */}
      <div className="flex flex-col flex-1 min-w-0">
        <div className="shrink-0 px-5 py-3 border-b border-zinc-800 flex items-center justify-between bg-zinc-900/50">
          <div className="flex items-center gap-3">
            <Link
              href="/agents"
              className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
            </Link>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-medium">{agent.name}</h1>
                <PillBadge className="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-400 font-medium">
                  Scheduler
                </PillBadge>
              </div>
              <p className="text-xs text-zinc-500">
                {agent.model_provider}/{agent.model_name}
                {sessionId && (
                  <span className="ml-2 font-mono">
                    {sessionId.slice(0, 8)}
                  </span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isStreaming && (
              <button
                onClick={handleCancel}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-900/30 text-red-400 text-xs font-medium hover:bg-red-900/50 transition-colors"
              >
                <XCircle className="w-3.5 h-3.5" />
                Cancel
              </button>
            )}
            <button
              onClick={() => setShowSessionPanel((v) => !v)}
              className={`p-1.5 rounded transition-colors ${
                showSessionPanel
                  ? "bg-zinc-700 text-white"
                  : "hover:bg-zinc-800 text-zinc-500"
              }`}
              title="Toggle session panel"
            >
              <PanelRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto px-5 py-4 space-y-4">
          {messages.length === 0 && (
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
            disabled={isStreaming}
          />
        </div>
      </div>

      {/* Middle: Session Panel (togglable) */}
      {showSessionPanel && (
        <div className="w-72 shrink-0 border-l border-zinc-800 bg-zinc-950/50">
          <SessionPanel
            agentId={agentId}
            scopeId={agentId}
            currentSessionId={sessionId}
            onSwitch={handleSessionSwitch}
            onCreated={handleSessionCreated}
            onForked={handleSessionForked}
          />
        </div>
      )}

      {/* Right: Orchestration Panel */}
      <div className="w-80 shrink-0 border-l border-zinc-800 flex flex-col overflow-hidden bg-zinc-950/50">
        <div className="px-4 py-3 border-b border-zinc-800">
          <h2 className="text-xs font-medium text-zinc-400 uppercase mb-2">
            Orchestration
          </h2>
          <div className="flex items-center gap-2">
            <StatusIcon
              className={`w-4 h-4 ${statusConfig[orchestrationStatus].color} ${orchestrationStatus === "running" ? "animate-spin" : ""}`}
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
            <p className="text-[10px] text-zinc-600 mt-1 font-mono truncate">
              State: {stateId}
            </p>
          )}
        </div>

        <div className="flex-1 overflow-auto">
          <div className="px-4 py-3">
            <h2 className="text-xs font-medium text-zinc-400 uppercase mb-2">
              Child Agents
              {children.length > 0 && (
                <span className="ml-1 text-zinc-500">({children.length})</span>
              )}
            </h2>

            {children.length === 0 && (
              <p className="text-xs text-zinc-600">
                {orchestrationStatus === "idle"
                  ? "No orchestration running"
                  : "No child agents spawned yet"}
              </p>
            )}

            <div className="space-y-2">
              {children.map((child) => (
                <div
                  key={child.id}
                  className="rounded-lg bg-zinc-900/50 border border-zinc-800"
                >
                  <button
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
                          {child.id.length > 20
                            ? "..." + child.id.slice(-16)
                            : child.id}
                        </span>
                        <PillBadge
                          className={`text-[10px] px-1.5 py-0.5 rounded font-medium shrink-0 ${childStatusBadge[child.status] || "bg-zinc-700 text-zinc-300"}`}
                        >
                          {child.status}
                        </PillBadge>
                      </div>
                      <UserInputCompact input={child.task} maxLength={60} />
                    </div>
                  </button>

                  {expandedChildren.has(child.id) && (
                    <div className="px-3 pb-2 border-t border-zinc-800">
                      {child.result_summary && (
                        <div className="mt-2 text-xs text-zinc-400 whitespace-pre-wrap max-h-32 overflow-auto">
                          <span className="text-zinc-500 font-medium">
                            Result:{" "}
                          </span>
                          {child.result_summary.slice(0, 500)}
                        </div>
                      )}
                      {(childMessages[child.id] || []).length > 0 && (
                        <div className="mt-2 space-y-1.5 max-h-48 overflow-auto">
                          {(childMessages[child.id] || []).map((cm) => (
                            <div key={cm.id} className="text-[11px]">
                              <span
                                className={`font-medium ${cm.role === "assistant" ? "text-green-400" : cm.role === "tool" ? "text-amber-400" : "text-zinc-400"}`}
                              >
                                {cm.role}
                                {cm.name ? ` — ${cm.name}` : ""}:
                              </span>{" "}
                              <span className="text-zinc-400 whitespace-pre-wrap">
                                {cm.content.slice(0, 300)}
                                {cm.content.length > 300 ? "..." : ""}
                              </span>
                              {cm.isStreaming && (
                                <Loader2 className="inline w-2.5 h-2.5 ml-1 animate-spin text-zinc-500" />
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      {!child.result_summary &&
                        (childMessages[child.id] || []).length === 0 && (
                          <p className="mt-2 text-[10px] text-zinc-600">
                            No output yet
                          </p>
                        )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
