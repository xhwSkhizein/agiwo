"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, PanelRight } from "lucide-react";
import { ChatInputBar } from "@/components/chat-input-bar";
import { ChatMessageItem } from "@/components/chat-message";
import { EmptyStateMessage, FullPageMessage } from "@/components/state-message";
import { SessionPanel } from "@/components/session-panel/session-panel";
import { getAgent, chatStreamUrl, getSessionSteps } from "@/lib/api";
import type { AgentConfig, StepResponse } from "@/lib/api";
import { useChatStream } from "@/hooks/use-chat-stream";
import type { ChatMessage } from "@/lib/chat-types";
import { genMessageId } from "@/lib/chat-types";

function stepsToMessages(steps: StepResponse[]): ChatMessage[] {
  const msgs: ChatMessage[] = [];
  for (const step of steps) {
    if (step.role === "user" && step.user_input) {
      const text =
        typeof step.user_input === "string"
          ? step.user_input
          : JSON.stringify(step.user_input);
      if (text) {
        msgs.push({ id: genMessageId(), role: "user", content: text });
      }
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

export default function AgentChatPage() {
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
  const [showSessionPanel, setShowSessionPanel] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const updateSessionUrl = useCallback(
    (sid: string) => {
      const url = new URL(window.location.href);
      url.searchParams.set("session", sid);
      router.replace(url.pathname + url.search);
    },
    [router],
  );

  const {
    messages,
    isStreaming,
    sendMessage,
    clearMessages,
    loadHistoryMessages,
  } = useChatStream(chatStreamUrl(agentId), {
    onSessionCaptured: (sid) => {
      if (sessionId !== sid) {
        setSessionId(sid);
        updateSessionUrl(sid);
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
          if (history.length > 0) {
            loadHistoryMessages(history);
          }
        })
        .catch(() => {});
    }
  }, [sessionId, messages.length, isStreaming, loadHistoryMessages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");
    sendMessage(text, sessionId);
  };

  const handleSessionSwitch = async (targetSessionId: string) => {
    setSessionId(targetSessionId);
    updateSessionUrl(targetSessionId);
    clearMessages();
    try {
      const steps = await getSessionSteps(targetSessionId);
      loadHistoryMessages(stepsToMessages(steps));
    } catch {}
  };

  const handleSessionCreated = (newSessionId: string) => {
    setSessionId(newSessionId);
    updateSessionUrl(newSessionId);
    clearMessages();
  };

  const handleSessionForked = async (forkedSessionId: string) => {
    setSessionId(forkedSessionId);
    updateSessionUrl(forkedSessionId);
    clearMessages();
  };

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
              className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
            </Link>
            <div>
              <h1 className="text-sm font-medium">{agent.name}</h1>
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
            onSubmit={handleSend}
            disabled={isStreaming}
          />
        </div>
      </div>

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
    </div>
  );
}
