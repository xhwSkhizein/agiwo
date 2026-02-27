"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Send,
  User,
  Bot,
  Wrench,
  Loader2,
  XCircle,
  Clock,
  CheckCircle2,
  AlertCircle,
  Moon,
  Play,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import {
  getAgent,
  schedulerChatStreamUrl,
  cancelSchedulerChat,
  getAgentStateChildren,
} from "@/lib/api";
import type { AgentConfig, AgentStateListItem } from "@/lib/api";

function genId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool" | "system";
  content: string;
  name?: string;
  tool_calls?: Record<string, unknown>[];
  reasoning_content?: string;
  isStreaming?: boolean;
  agentId?: string;
}

type OrchestrationStatus = "idle" | "running" | "sleeping" | "completed" | "failed" | "cancelled";

const statusConfig: Record<OrchestrationStatus, { icon: typeof Play; color: string; label: string }> = {
  idle: { icon: Clock, color: "text-zinc-500", label: "Idle" },
  running: { icon: Loader2, color: "text-blue-400", label: "Running" },
  sleeping: { icon: Moon, color: "text-yellow-400", label: "Sleeping" },
  completed: { icon: CheckCircle2, color: "text-green-400", label: "Completed" },
  failed: { icon: AlertCircle, color: "text-red-400", label: "Failed" },
  cancelled: { icon: XCircle, color: "text-zinc-400", label: "Cancelled" },
};

const childStatusBadge: Record<string, string> = {
  pending: "bg-zinc-700 text-zinc-300",
  running: "bg-blue-900/50 text-blue-400",
  sleeping: "bg-yellow-900/50 text-yellow-400",
  completed: "bg-green-900/50 text-green-400",
  failed: "bg-red-900/50 text-red-400",
};

export default function SchedulerChatPage() {
  const params = useParams();
  const agentId = params.id as string;

  const [agent, setAgent] = useState<AgentConfig | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  const [orchestrationStatus, setOrchestrationStatus] = useState<OrchestrationStatus>("idle");
  const [stateId, setStateId] = useState<string | null>(null);
  const [children, setChildren] = useState<AgentStateListItem[]>([]);
  const [expandedChildren, setExpandedChildren] = useState<Set<string>>(new Set());
  const [childMessages, setChildMessages] = useState<Record<string, ChatMessage[]>>({});
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    getAgent(agentId)
      .then(setAgent)
      .catch(() => setAgent(null))
      .finally(() => setLoading(false));
  }, [agentId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!startTime || orchestrationStatus === "completed" || orchestrationStatus === "failed") return;
    const timer = setInterval(() => {
      setElapsed(((Date.now() - startTime) / 1000));
    }, 100);
    return () => clearInterval(timer);
  }, [startTime, orchestrationStatus]);

  const pollChildren = useCallback((sid: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const result = await getAgentStateChildren(sid);
        setChildren(result);
      } catch {
        // ignore polling errors
      }
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

  const handleCancel = async () => {
    if (!stateId) return;
    try {
      await cancelSchedulerChat(agentId, stateId);
      setOrchestrationStatus("cancelled");
      stopPolling();
      setMessages((prev) => [
        ...prev,
        { id: genId(), role: "system", content: "Orchestration cancelled by user." },
      ]);
    } catch (err) {
      console.error("Cancel failed:", err);
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

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    setInput("");
    setIsStreaming(true);
    setOrchestrationStatus("running");
    setStartTime(Date.now());
    setChildren([]);
    setChildMessages({});

    const userMsg: ChatMessage = { id: genId(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    const assistantMsg: ChatMessage = {
      id: genId(),
      role: "assistant",
      content: "",
      isStreaming: true,
    };
    setMessages((prev) => [...prev, assistantMsg]);

    let currentAssistantContent = "";
    let currentReasoningContent = "";
    let rootAgentId: string | null = null;

    try {
      const res = await fetch(schedulerChatStreamUrl(agentId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error("No response body");

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const dataStr = line.slice(5).trim();
          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);
            const eventAgentId = data.agent_id || null;

            if (data.type === "run_started" && data.data?.session_id) {
              const capturedSessionId = data.data.session_id;
              if (!sessionId) setSessionId(capturedSessionId);
              if (!rootAgentId) {
                rootAgentId = eventAgentId;
                setStateId(eventAgentId);
                if (eventAgentId) pollChildren(eventAgentId);
              }
            }

            const isChildEvent = rootAgentId && eventAgentId && eventAgentId !== rootAgentId;

            if (isChildEvent) {
              handleChildEvent(eventAgentId, data);
              continue;
            }

            if (data.type === "step_delta" && data.delta) {
              if (data.delta.content) {
                currentAssistantContent += data.delta.content;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsg.id
                      ? { ...m, content: currentAssistantContent, reasoning_content: currentReasoningContent || undefined }
                      : m
                  )
                );
              }
              if (data.delta.reasoning_content) {
                currentReasoningContent += data.delta.reasoning_content;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsg.id
                      ? { ...m, reasoning_content: currentReasoningContent }
                      : m
                  )
                );
              }
            }

            if (data.type === "step_completed" && data.step) {
              const step = data.step;
              if (step.role === "assistant") {
                const tc = step.tool_calls;
                if (tc && tc.length > 0) {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMsg.id
                        ? { ...m, tool_calls: tc, isStreaming: false }
                        : m
                    )
                  );

                  for (const call of tc) {
                    const fn = call.function as Record<string, unknown> | undefined;
                    const toolName = fn?.name as string | undefined;
                    if (toolName === "sleep_and_wait") {
                      setOrchestrationStatus("sleeping");
                    }
                  }
                }
              }
              if (step.role === "tool") {
                const toolMsg: ChatMessage = {
                  id: genId(),
                  role: "tool",
                  content: typeof step.content === "string" ? step.content : JSON.stringify(step.content),
                  name: step.name || undefined,
                };
                setMessages((prev) => [...prev, toolMsg]);

                const nextAssistant: ChatMessage = {
                  id: genId(),
                  role: "assistant",
                  content: "",
                  isStreaming: true,
                };
                assistantMsg.id = nextAssistant.id;
                currentAssistantContent = "";
                currentReasoningContent = "";
                setMessages((prev) => [...prev, nextAssistant]);
              }
            }

            if (data.type === "run_completed") {
              setMessages((prev) =>
                prev
                  .map((m) =>
                    m.id === assistantMsg.id ? { ...m, isStreaming: false } : m
                  )
                  .filter((m) => !(m.role === "assistant" && !m.content && !m.tool_calls && !m.isStreaming))
              );
              if (orchestrationStatus === "sleeping") {
                // still sleeping, run_completed is just for one agent.run()
              } else {
                setOrchestrationStatus("running");
              }
            }

            if (data.type === "run_started" && rootAgentId && eventAgentId === rootAgentId) {
              setOrchestrationStatus("running");
              if (!currentAssistantContent && assistantMsg.id) {
                // woke up, start fresh assistant message
                const wakeMsg: ChatMessage = {
                  id: genId(),
                  role: "system",
                  content: "Agent woke up — resuming execution",
                };
                const nextAssistant: ChatMessage = {
                  id: genId(),
                  role: "assistant",
                  content: "",
                  isStreaming: true,
                };
                assistantMsg.id = nextAssistant.id;
                currentAssistantContent = "";
                currentReasoningContent = "";
                setMessages((prev) => [
                  ...prev.filter((m) => !(m.role === "assistant" && !m.content && !m.tool_calls)),
                  wakeMsg,
                  nextAssistant,
                ]);
              }
            }

            if (data.type === "scheduler_completed") {
              setOrchestrationStatus("completed");
              stopPolling();
              if (stateId) {
                try {
                  const finalChildren = await getAgentStateChildren(stateId);
                  setChildren(finalChildren);
                } catch { /* ignore */ }
              }
            }

            if (data.type === "scheduler_failed") {
              setOrchestrationStatus("failed");
              stopPolling();
              setMessages((prev) => [
                ...prev.filter((m) => !(m.role === "assistant" && !m.content && !m.tool_calls)),
                { id: genId(), role: "system", content: `Orchestration failed: ${data.error || "Unknown error"}` },
              ]);
            }
          } catch {
            // skip non-JSON lines
          }
        }
      }
    } catch (err) {
      setOrchestrationStatus("failed");
      stopPolling();
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id
            ? { ...m, content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`, isStreaming: false }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  const handleChildEvent = (childAgentId: string, data: Record<string, unknown>) => {
    const type = data.type as string;
    if (type === "step_delta" && data.delta) {
      const delta = data.delta as Record<string, string>;
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
                  : m
              ),
            };
          }
          return {
            ...prev,
            [childAgentId]: [
              ...msgs,
              { id: genId(), role: "assistant" as const, content: delta.content, isStreaming: true, agentId: childAgentId },
            ],
          };
        });
      }
    }
    if (type === "step_completed" && data.step) {
      const step = data.step as Record<string, unknown>;
      if (step.role === "tool") {
        const toolMsg: ChatMessage = {
          id: genId(),
          role: "tool",
          content: typeof step.content === "string" ? step.content : JSON.stringify(step.content),
          name: (step.name as string) || undefined,
          agentId: childAgentId,
        };
        setChildMessages((prev) => ({
          ...prev,
          [childAgentId]: [...(prev[childAgentId] || []), toolMsg],
        }));
      }
    }
    if (type === "run_completed") {
      setChildMessages((prev) => {
        const msgs = prev[childAgentId] || [];
        return {
          ...prev,
          [childAgentId]: msgs.map((m) =>
            m.isStreaming ? { ...m, isStreaming: false } : m
          ),
        };
      });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Loading agent...</div>
      </div>
    );
  }

  if (!agent) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Agent not found</div>
      </div>
    );
  }

  const StatusIcon = statusConfig[orchestrationStatus].icon;

  return (
    <div className="flex h-full">
      {/* Left: Chat */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <div className="shrink-0 px-5 py-3 border-b border-zinc-800 flex items-center justify-between bg-zinc-900/50">
          <div className="flex items-center gap-3">
            <Link href="/agents" className="p-1.5 rounded hover:bg-zinc-800 transition-colors">
              <ArrowLeft className="w-4 h-4" />
            </Link>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-medium">{agent.name}</h1>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-400 font-medium">
                  Scheduler
                </span>
              </div>
              <p className="text-xs text-zinc-500">
                {agent.model_provider}/{agent.model_name}
                {sessionId && <span className="ml-2 font-mono">{sessionId.slice(0, 8)}</span>}
              </p>
            </div>
          </div>
          {isStreaming && (
            <button
              onClick={handleCancel}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-900/30 text-red-400 text-xs font-medium hover:bg-red-900/50 transition-colors"
            >
              <XCircle className="w-3.5 h-3.5" />
              Cancel
            </button>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-auto px-5 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full text-zinc-600 text-sm">
              Send a message to start the scheduler orchestration
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className="flex gap-3">
              <div className="shrink-0 mt-1">
                {msg.role === "user" && (
                  <div className="w-7 h-7 rounded-full bg-blue-900/50 flex items-center justify-center">
                    <User className="w-3.5 h-3.5 text-blue-400" />
                  </div>
                )}
                {msg.role === "assistant" && (
                  <div className="w-7 h-7 rounded-full bg-green-900/50 flex items-center justify-center">
                    <Bot className="w-3.5 h-3.5 text-green-400" />
                  </div>
                )}
                {msg.role === "tool" && (
                  <div className="w-7 h-7 rounded-full bg-amber-900/50 flex items-center justify-center">
                    <Wrench className="w-3.5 h-3.5 text-amber-400" />
                  </div>
                )}
                {msg.role === "system" && (
                  <div className="w-7 h-7 rounded-full bg-purple-900/50 flex items-center justify-center">
                    <Clock className="w-3.5 h-3.5 text-purple-400" />
                  </div>
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-zinc-400 uppercase">
                    {msg.role}
                    {msg.name && ` — ${msg.name}`}
                  </span>
                  {msg.isStreaming && <Loader2 className="w-3 h-3 text-zinc-500 animate-spin" />}
                </div>

                {msg.reasoning_content && (
                  <div className="mb-2 px-3 py-2 rounded bg-zinc-800/50 text-xs text-zinc-400 whitespace-pre-wrap max-h-32 overflow-auto">
                    <span className="text-zinc-500 font-medium">Thinking: </span>
                    {msg.reasoning_content}
                  </div>
                )}

                {msg.content && (
                  <div className={`text-sm whitespace-pre-wrap break-words ${msg.role === "system" ? "text-purple-300 italic" : ""}`}>
                    {msg.content}
                  </div>
                )}

                {msg.tool_calls && msg.tool_calls.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {msg.tool_calls.map((tc, i) => {
                      const fn = tc.function as Record<string, unknown> | undefined;
                      return (
                        <div key={i} className="text-xs bg-zinc-800/50 rounded px-3 py-2 font-mono overflow-auto max-h-32">
                          <span className="text-amber-400">{fn ? (fn.name as string) : "tool_call"}</span>
                          <span className="text-zinc-500 ml-2">{fn ? (fn.arguments as string) : JSON.stringify(tc)}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="shrink-0 px-5 py-3 border-t border-zinc-800 bg-zinc-900/50">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              sendMessage();
            }}
            className="flex items-center gap-3"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type a message..."
              disabled={isStreaming}
              className="flex-1 px-4 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-sm focus:outline-none focus:border-zinc-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="p-2.5 rounded-lg bg-white text-black hover:bg-zinc-200 transition-colors disabled:opacity-30"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>

      {/* Right: Orchestration Panel */}
      <div className="w-80 shrink-0 border-l border-zinc-800 flex flex-col overflow-hidden bg-zinc-950/50">
        {/* Status */}
        <div className="px-4 py-3 border-b border-zinc-800">
          <h2 className="text-xs font-medium text-zinc-400 uppercase mb-2">Orchestration</h2>
          <div className="flex items-center gap-2">
            <StatusIcon
              className={`w-4 h-4 ${statusConfig[orchestrationStatus].color} ${orchestrationStatus === "running" ? "animate-spin" : ""}`}
            />
            <span className={`text-sm font-medium ${statusConfig[orchestrationStatus].color}`}>
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

        {/* Children */}
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
                <div key={child.id} className="rounded-lg bg-zinc-900/50 border border-zinc-800">
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
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium shrink-0 ${childStatusBadge[child.status] || "bg-zinc-700 text-zinc-300"}`}>
                          {child.status}
                        </span>
                      </div>
                      <p className="text-[10px] text-zinc-500 truncate mt-0.5">
                        {child.task.slice(0, 60)}{child.task.length > 60 ? "..." : ""}
                      </p>
                    </div>
                  </button>

                  {expandedChildren.has(child.id) && (
                    <div className="px-3 pb-2 border-t border-zinc-800">
                      {child.result_summary && (
                        <div className="mt-2 text-xs text-zinc-400 whitespace-pre-wrap max-h-32 overflow-auto">
                          <span className="text-zinc-500 font-medium">Result: </span>
                          {child.result_summary.slice(0, 500)}
                        </div>
                      )}
                      {(childMessages[child.id] || []).length > 0 && (
                        <div className="mt-2 space-y-1.5 max-h-48 overflow-auto">
                          {(childMessages[child.id] || []).map((cm) => (
                            <div key={cm.id} className="text-[11px]">
                              <span className={`font-medium ${cm.role === "assistant" ? "text-green-400" : cm.role === "tool" ? "text-amber-400" : "text-zinc-400"}`}>
                                {cm.role}{cm.name ? ` — ${cm.name}` : ""}:
                              </span>{" "}
                              <span className="text-zinc-400 whitespace-pre-wrap">
                                {cm.content.slice(0, 300)}{cm.content.length > 300 ? "..." : ""}
                              </span>
                              {cm.isStreaming && <Loader2 className="inline w-2.5 h-2.5 ml-1 animate-spin text-zinc-500" />}
                            </div>
                          ))}
                        </div>
                      )}
                      {!child.result_summary && (childMessages[child.id] || []).length === 0 && (
                        <p className="mt-2 text-[10px] text-zinc-600">No output yet</p>
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
