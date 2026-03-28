"use client";

import { useCallback, useRef, useState } from "react";
import { parseStreamEventPayload } from "@/lib/api";
import type {
  AgentStreamEventPayload,
  RunCompletedEventPayload,
  StreamEventPayload,
  StepResponse,
} from "@/lib/api";
import type { ChatMessage } from "@/lib/chat-types";
import { genMessageId } from "@/lib/chat-types";

export interface ChatStreamCallbacks {
  onSessionCaptured?: (sessionId: string) => void;
  onRootAgentCaptured?: (agentId: string) => void;
  onChildEvent?: (agentId: string, event: StreamEventPayload) => void;
  onSchedulerFailed?: (error: string) => void;
  onRunCompleted?: (event: RunCompletedEventPayload) => void;
  onRunStarted?: (event: AgentStreamEventPayload) => void;
}

export function useChatStream(
  streamUrl: string,
  callbacks: ChatStreamCallbacks = {},
) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  const loadHistoryMessages = useCallback((history: ChatMessage[]) => {
    setMessages(history);
  }, []);

  const sendMessage = useCallback(
    async (text: string, sessionId: string | null) => {
      if (!text.trim() || isStreaming) return;

      setIsStreaming(true);

      const userMsg: ChatMessage = {
        id: genMessageId(),
        role: "user",
        content: text,
      };

      const assistantMsg: ChatMessage = {
        id: genMessageId(),
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      let currentAssistantContent = "";
      let currentReasoningContent = "";
      let currentAssistantId = assistantMsg.id;
      let rootAgentId: string | null = null;

      try {
        const res = await fetch(streamUrl, {
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
              const data = parseStreamEventPayload(dataStr);
              if (!data) continue;

              if (data.type === "scheduler_failed") {
                callbacksRef.current.onSchedulerFailed?.(
                  "error" in data ? String(data.error) : "Unknown error",
                );
                continue;
              }

              const agentEvent = data as AgentStreamEventPayload;
              const eventAgentId =
                "agent_id" in data && typeof data.agent_id === "string"
                  ? data.agent_id
                  : null;

              if (agentEvent.type === "run_started" && agentEvent.session_id) {
                callbacksRef.current.onSessionCaptured?.(agentEvent.session_id);
                if (!rootAgentId && eventAgentId) {
                  rootAgentId = eventAgentId;
                  callbacksRef.current.onRootAgentCaptured?.(eventAgentId);
                } else if (rootAgentId && eventAgentId === rootAgentId) {
                  if (!currentAssistantContent) {
                    const wakeMsg: ChatMessage = {
                      id: genMessageId(),
                      role: "system",
                      content: "Agent woke up — resuming execution",
                    };
                    const nextAssistant: ChatMessage = {
                      id: genMessageId(),
                      role: "assistant",
                      content: "",
                      isStreaming: true,
                    };
                    currentAssistantId = nextAssistant.id;
                    currentAssistantContent = "";
                    currentReasoningContent = "";
                    setMessages((prev) => [
                      ...prev.filter(
                        (m) =>
                          !(
                            m.role === "assistant" &&
                            !m.content &&
                            !m.tool_calls
                          ),
                      ),
                      wakeMsg,
                      nextAssistant,
                    ]);
                  }
                }
                callbacksRef.current.onRunStarted?.(agentEvent);
              }

              const isChildEvent =
                rootAgentId && eventAgentId && eventAgentId !== rootAgentId;
              if (isChildEvent) {
                callbacksRef.current.onChildEvent?.(eventAgentId, data);
                continue;
              }

              if (agentEvent.type === "step_delta" && "delta" in agentEvent && agentEvent.delta) {
                if (agentEvent.delta.content) {
                  currentAssistantContent += agentEvent.delta.content;
                  const snapshot = currentAssistantContent;
                  const reasoningSnapshot = currentReasoningContent;
                  const aid = currentAssistantId;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === aid
                        ? {
                            ...m,
                            content: snapshot,
                            reasoning_content: reasoningSnapshot || undefined,
                          }
                        : m,
                    ),
                  );
                }
                if (agentEvent.delta.reasoning_content) {
                  currentReasoningContent += agentEvent.delta.reasoning_content;
                  const snapshot = currentReasoningContent;
                  const aid = currentAssistantId;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === aid
                        ? { ...m, reasoning_content: snapshot }
                        : m,
                    ),
                  );
                }
              }

              if (agentEvent.type === "step_completed" && "step" in agentEvent && agentEvent.step) {
                const step = agentEvent.step as StepResponse;
                if (step.role === "assistant") {
                  const tc = step.tool_calls;
                  if (tc && tc.length > 0) {
                    const aid = currentAssistantId;
                    setMessages((prev) =>
                      prev.map((m) =>
                        m.id === aid
                          ? { ...m, tool_calls: tc, isStreaming: false }
                          : m,
                      ),
                    );
                  }
                }
                if (step.role === "tool") {
                  const toolMsg: ChatMessage = {
                    id: genMessageId(),
                    role: "tool",
                    content:
                      typeof step.content === "string"
                        ? step.content
                        : JSON.stringify(step.content),
                    name: step.name || undefined,
                  };
                  const nextAssistant: ChatMessage = {
                    id: genMessageId(),
                    role: "assistant",
                    content: "",
                    isStreaming: true,
                  };
                  currentAssistantId = nextAssistant.id;
                  currentAssistantContent = "";
                  currentReasoningContent = "";
                  setMessages((prev) => [...prev, toolMsg, nextAssistant]);
                }
              }

              if (agentEvent.type === "run_completed") {
                const aid = currentAssistantId;
                setMessages((prev) =>
                  prev
                    .map((m) =>
                      m.id === aid ? { ...m, isStreaming: false } : m,
                    )
                    .filter(
                      (m) =>
                        !(
                          m.role === "assistant" &&
                          !m.content &&
                          !m.tool_calls &&
                          !m.isStreaming
                        ),
                    ),
                );
                callbacksRef.current.onRunCompleted?.(
                  agentEvent as RunCompletedEventPayload,
                );
              }

              if (agentEvent.type === "run_failed") {
                const aid = currentAssistantId;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === aid
                      ? {
                          ...m,
                          content: `Error: ${"error" in agentEvent ? agentEvent.error : "Unknown error"}`,
                          isStreaming: false,
                        }
                      : m,
                  ),
                );
              }
            } catch {
              // skip non-JSON lines
            }
          }
        }
      } catch (err) {
        const aid = currentAssistantId;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aid
              ? {
                  ...m,
                  content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
                  isStreaming: false,
                }
              : m,
          ),
        );
      } finally {
        setIsStreaming(false);
      }
    },
    [streamUrl, isStreaming],
  );

  return { messages, isStreaming, sendMessage, clearMessages, loadHistoryMessages };
}
