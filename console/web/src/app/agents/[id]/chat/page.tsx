"use client";

import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { ChatInputBar } from "@/components/chat-input-bar";
import { ChatMessageItem } from "@/components/chat-message";
import { EmptyStateMessage, FullPageMessage } from "@/components/state-message";
import { getAgent, chatStreamUrl, parseStreamEventPayload } from "@/lib/api";
import type { AgentConfig, AgentStreamEventPayload, ToolCallPayload } from "@/lib/api";

function genId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  name?: string;
  tool_calls?: ToolCallPayload[];
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
      const pendingToolMessages: ChatMessage[] = [];
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
            const data = parseStreamEventPayload(dataStr);
            if (!data) {
              continue;
            }
            if (data.type === "scheduler_completed" || data.type === "scheduler_failed") {
              continue;
            }
            const agentEvent = data as AgentStreamEventPayload;

            if (agentEvent.type === "run_started" && agentEvent.data?.session_id) {
              capturedSessionId = String(agentEvent.data.session_id);
              if (!sessionId) setSessionId(capturedSessionId);
            }

            if (agentEvent.type === "step_delta" && agentEvent.delta) {
              if (agentEvent.delta.content) {
                currentAssistantContent += agentEvent.delta.content;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsg.id
                      ? { ...m, content: currentAssistantContent, reasoning_content: currentReasoningContent || undefined }
                      : m
                  )
                );
              }
              if (agentEvent.delta.reasoning_content) {
                currentReasoningContent += agentEvent.delta.reasoning_content;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsg.id
                      ? { ...m, reasoning_content: currentReasoningContent }
                      : m
                  )
                );
              }
            }

            if (agentEvent.type === "step_completed" && agentEvent.step) {
              const step = agentEvent.step;
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

            if (agentEvent.type === "run_completed") {
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
    return <FullPageMessage>Loading agent...</FullPageMessage>;
  }

  if (!agent) {
    return <FullPageMessage>Agent not found</FullPageMessage>;
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
          <EmptyStateMessage className="flex items-center justify-center h-full text-zinc-600 text-sm">
            Send a message to start the conversation
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
          onSubmit={sendMessage}
          disabled={isStreaming}
        />
      </div>
    </div>
  );
}
