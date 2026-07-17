"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { parseStreamEventPayload } from "@/lib/api";
import type {
  AgentStreamEventPayload,
  RunCompletedEventPayload,
  StreamEventPayload,
  StepResponse,
} from "@/lib/api";
import type { ChatMessage } from "@/lib/chat-types";
import { contentToText, genMessageId } from "@/lib/chat-types";

export interface ChatStreamCallbacks {
  onSessionCaptured?: (sessionId: string) => void;
  onRootStateCaptured?: (stateId: string) => void;
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
  const abortRef = useRef<AbortController | null>(null);
  const streamVersionRef = useRef(0);
  callbacksRef.current = callbacks;

  const abortStream = useCallback(() => {
    streamVersionRef.current += 1;
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  useEffect(() => abortStream, [abortStream]);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  const loadHistoryMessages = useCallback((history: ChatMessage[]) => {
    setMessages(history);
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isStreaming || !streamUrl) return;

      abortStream();
      const controller = new AbortController();
      abortRef.current = controller;
      streamVersionRef.current += 1;
      const streamVersion = streamVersionRef.current;
      setIsStreaming(true);

      const userMsg: ChatMessage = {
        id: genMessageId(),
        role: "user",
        text,
        userInput: text,
        rawContent: text,
      };

      const assistantMsg: ChatMessage = {
        id: genMessageId(),
        role: "assistant",
        text: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      let currentAssistantText = "";
      let currentReasoningText = "";
      let currentAssistantId: string | null = assistantMsg.id;
      let rootStateId: string | null = null;
      let hasSeenRootRunStarted = false;

      const appendAssistantPlaceholder = () => {
        const nextAssistant: ChatMessage = {
          id: genMessageId(),
          role: "assistant",
          text: "",
          isStreaming: true,
        };
        currentAssistantId = nextAssistant.id;
        currentAssistantText = "";
        currentReasoningText = "";
        setMessages((prev) => [...prev, nextAssistant]);
        return nextAssistant.id;
      };

      const ensureAssistantPlaceholder = () =>
        currentAssistantId ?? appendAssistantPlaceholder();

      const finishCurrentAssistant = (dropIfEmpty: boolean) => {
        if (!currentAssistantId) {
          return;
        }
        const assistantId = currentAssistantId;
        setMessages((prev) =>
          prev.flatMap((message) => {
            if (message.id !== assistantId) {
              return [message];
            }
            const hasVisibleContent = Boolean(
              message.text ||
                message.toolCalls?.length ||
                message.reasoningContent ||
                message.rawContent,
            );
            if (dropIfEmpty && !hasVisibleContent) {
              return [];
            }
            return [{ ...message, isStreaming: false }];
          }),
        );
        currentAssistantId = null;
        currentAssistantText = "";
        currentReasoningText = "";
      };

      try {
        const res = await fetch(streamUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
          signal: controller.signal,
        });

        if (!res.ok) {
          throw new Error(describeStreamError(res.status));
        }

        const reader = res.body?.getReader();
        const decoder = new TextDecoder();
        if (!reader) throw new Error("No response body");

        let buffer = "";
        while (true) {
          if (streamVersionRef.current !== streamVersion) {
            return;
          }
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;

            const data = parseStreamEventPayload(dataStr);
            if (!data) continue;

            if (data.type === "scheduler_failed") {
              callbacksRef.current.onSchedulerFailed?.(
                "error" in data ? String(data.error) : "Unknown error",
              );
              continue;
            }

            if (data.type === "scheduler_ack") {
              if (data.state_id) {
                rootStateId = data.state_id;
                callbacksRef.current.onRootStateCaptured?.(data.state_id);
              }
              if (data.session_id) {
                callbacksRef.current.onSessionCaptured?.(data.session_id);
              }
              continue;
            }

            const agentEvent = data as AgentStreamEventPayload;
            const eventAgentId =
              "agent_id" in data && typeof data.agent_id === "string"
                ? data.agent_id
                : null;

            if (agentEvent.type === "run_started" && agentEvent.session_id) {
              callbacksRef.current.onSessionCaptured?.(agentEvent.session_id);
              if (!rootStateId && eventAgentId) {
                rootStateId = eventAgentId;
                callbacksRef.current.onRootStateCaptured?.(eventAgentId);
              }
              if (hasSeenRootRunStarted) {
                appendAssistantPlaceholder();
              } else {
                hasSeenRootRunStarted = true;
              }
              callbacksRef.current.onRunStarted?.(agentEvent);
            }

            const isChildEvent =
              rootStateId && eventAgentId && eventAgentId !== rootStateId;
            if (isChildEvent) {
              callbacksRef.current.onChildEvent?.(eventAgentId, data);
              continue;
            }

            if (agentEvent.type === "step_delta" && agentEvent.delta) {
              const assistantId = ensureAssistantPlaceholder();
              if (agentEvent.delta.content) {
                currentAssistantText += agentEvent.delta.content;
                const textSnapshot = currentAssistantText;
                const reasoningSnapshot = currentReasoningText;
                setMessages((prev) =>
                  prev.map((message) =>
                    message.id === assistantId
                      ? {
                          ...message,
                          text: textSnapshot,
                          reasoningContent: reasoningSnapshot || undefined,
                        }
                      : message,
                  ),
                );
              }
              if (agentEvent.delta.reasoning_content) {
                currentReasoningText += agentEvent.delta.reasoning_content;
                const reasoningSnapshot = currentReasoningText;
                setMessages((prev) =>
                  prev.map((message) =>
                    message.id === assistantId
                      ? { ...message, reasoningContent: reasoningSnapshot }
                      : message,
                  ),
                );
              }
            }

            if (agentEvent.type === "step_completed" && agentEvent.step) {
              const step = agentEvent.step as StepResponse;
              if (step.role === "assistant") {
                const assistantId = ensureAssistantPlaceholder();
                setMessages((prev) =>
                  prev.map((message) =>
                    message.id === assistantId
                      ? {
                          ...message,
                          stepId: step.id,
                          sequence: step.sequence,
                          text:
                            contentToText(step.content) ??
                            message.text ??
                            step.content_for_user ??
                            "",
                          rawContent: step.content ?? message.rawContent,
                          toolCalls: step.tool_calls ?? undefined,
                          reasoningContent:
                            step.reasoning_content ?? message.reasoningContent,
                          isStreaming: false,
                        }
                      : message,
                  ),
                );
                currentAssistantText =
                  contentToText(step.content) ?? currentAssistantText;
                currentReasoningText =
                  step.reasoning_content ?? currentReasoningText;
                currentAssistantId = null;
              }
              if (step.role === "tool") {
                finishCurrentAssistant(true);
                const hasCondensed = typeof step.condensed_content === "string";
                const toolMsg: ChatMessage = {
                  id: genMessageId(),
                  stepId: step.id,
                  role: "tool",
                  sequence: step.sequence,
                  text: hasCondensed
                    ? step.condensed_content!
                    : contentToText(step.content),
                  originalContent: hasCondensed
                    ? (contentToText(step.content) ?? undefined)
                    : undefined,
                  rawContent: step.content,
                  name: step.name || undefined,
                  sourceAgentId: step.agent_id ?? undefined,
                };
                setMessages((prev) => [...prev, toolMsg]);
              }
            }

            if (agentEvent.type === "run_completed") {
              finishCurrentAssistant(true);
              callbacksRef.current.onRunCompleted?.(
                agentEvent as RunCompletedEventPayload,
              );
            }

            if (agentEvent.type === "run_failed") {
              const assistantId = ensureAssistantPlaceholder();
              setMessages((prev) =>
                prev.map((message) =>
                  message.id === assistantId
                    ? {
                        ...message,
                        text: `Error: ${agentEvent.error ?? "Unknown error"}`,
                        isStreaming: false,
                      }
                    : message,
                ),
              );
              currentAssistantId = null;
            }
          }
        }
      } catch (err) {
        if (controller.signal.aborted) {
          return;
        }
        if (streamVersionRef.current !== streamVersion) {
          return;
        }
        const assistantId = ensureAssistantPlaceholder();
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  text: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
                  isStreaming: false,
                }
              : message,
          ),
        );
        currentAssistantId = null;
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        if (streamVersionRef.current === streamVersion) {
          setIsStreaming(false);
        }
      }
    },
    [abortStream, isStreaming, streamUrl],
  );

  return {
    messages,
    isStreaming,
    sendMessage,
    clearMessages,
    loadHistoryMessages,
    abortStream,
  };
}

function describeStreamError(status: number): string {
  if (status === 404) {
    return "Requested session or agent was not found.";
  }
  if (status === 422) {
    return "Request could not be processed by the scheduler.";
  }
  if (status >= 500) {
    return "Scheduler stream failed on the server.";
  }
  return `HTTP ${status}`;
}
