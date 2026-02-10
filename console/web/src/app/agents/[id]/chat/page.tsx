"use client";

import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Send, User, Bot, Wrench, Loader2 } from "lucide-react";
import { getAgent, getSessionSteps, chatStreamUrl } from "@/lib/api";
import type { AgentConfig, StepResponse } from "@/lib/api";

function genId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  name?: string;
  tool_calls?: Record<string, unknown>[];
  reasoning_content?: string;
  isStreaming?: boolean;
}

export default function AgentChatPage() {
  const params = useParams();
  const agentId = params.id as string;

  const [agent, setAgent] = useState<AgentConfig | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getAgent(agentId)
      .then(setAgent)
      .catch(() => setAgent(null))
      .finally(() => setLoading(false));
  }, [agentId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    const userMsg: ChatMessage = {
      id: genId(),
      role: "user",
      content: text,
    };
    setMessages((prev) => [...prev, userMsg]);

    const assistantMsg: ChatMessage = {
      id: genId(),
      role: "assistant",
      content: "",
      isStreaming: true,
    };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      const res = await fetch(chatStreamUrl(agentId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let currentAssistantContent = "";
      let currentReasoningContent = "";
      let pendingToolMessages: ChatMessage[] = [];
      let capturedSessionId = sessionId;

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
          if (!dataStr || dataStr === "") continue;

          try {
            const data = JSON.parse(dataStr);

            if (data.type === "run_started" && data.data?.session_id) {
              capturedSessionId = data.data.session_id;
              if (!sessionId) setSessionId(capturedSessionId);
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
                }
              }
              if (step.role === "tool") {
                const toolMsg: ChatMessage = {
                  id: genId(),
                  role: "tool",
                  content:
                    typeof step.content === "string"
                      ? step.content
                      : JSON.stringify(step.content),
                  name: step.name || undefined,
                };
                pendingToolMessages.push(toolMsg);
                setMessages((prev) => [...prev, toolMsg]);

                // New assistant message for next LLM response
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
                    m.id === assistantMsg.id
                      ? { ...m, isStreaming: false }
                      : m
                  )
                  .filter((m) => !(m.role === "assistant" && !m.content && !m.tool_calls && !m.isStreaming))
              );
            }
          } catch {
            // skip non-JSON lines
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id
            ? {
                ...m,
                content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
                isStreaming: false,
              }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
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

  return (
    <div className="flex flex-col h-full">
      <div className="shrink-0 px-5 py-3 border-b border-zinc-800 flex items-center gap-3 bg-zinc-900/50">
        <Link
          href="/agents"
          className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <h1 className="text-sm font-medium">{agent.name}</h1>
          <p className="text-xs text-zinc-500">
            {agent.model_provider}/{agent.model_name}
            {sessionId && (
              <span className="ml-2 font-mono">{sessionId.slice(0, 8)}</span>
            )}
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-auto px-5 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-zinc-600 text-sm">
            Send a message to start the conversation
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
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-zinc-400 uppercase">
                  {msg.role}
                  {msg.name && ` — ${msg.name}`}
                </span>
                {msg.isStreaming && (
                  <Loader2 className="w-3 h-3 text-zinc-500 animate-spin" />
                )}
              </div>

              {msg.reasoning_content && (
                <div className="mb-2 px-3 py-2 rounded bg-zinc-800/50 text-xs text-zinc-400 whitespace-pre-wrap max-h-32 overflow-auto">
                  <span className="text-zinc-500 font-medium">Thinking: </span>
                  {msg.reasoning_content}
                </div>
              )}

              {msg.content && (
                <div className="text-sm whitespace-pre-wrap break-words">
                  {msg.content}
                </div>
              )}

              {msg.tool_calls && msg.tool_calls.length > 0 && (
                <div className="mt-2 space-y-1">
                  {msg.tool_calls.map((tc, i) => {
                    const fn = tc.function as
                      | Record<string, unknown>
                      | undefined;
                    return (
                      <div
                        key={i}
                        className="text-xs bg-zinc-800/50 rounded px-3 py-2 font-mono overflow-auto max-h-32"
                      >
                        <span className="text-amber-400">
                          {fn ? (fn.name as string) : "tool_call"}
                        </span>
                        <span className="text-zinc-500 ml-2">
                          {fn
                            ? (fn.arguments as string)
                            : JSON.stringify(tc)}
                        </span>
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
  );
}
