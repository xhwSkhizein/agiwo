"use client";

import { Bot, Clock, Loader2, User, Wrench } from "lucide-react";
import type { ToolCallPayload } from "@/lib/api";

type ChatRole = "user" | "assistant" | "tool" | "system";

type ChatMessageLike = {
  role: ChatRole;
  content: string;
  name?: string;
  tool_calls?: ToolCallPayload[];
  reasoning_content?: string;
  isStreaming?: boolean;
};

const ROLE_STYLES: Record<
  ChatRole,
  {
    avatarClassName: string;
    iconClassName: string;
    contentClassName?: string;
    icon: typeof User;
  }
> = {
  user: {
    avatarClassName: "bg-blue-900/50",
    iconClassName: "text-blue-400",
    icon: User,
  },
  assistant: {
    avatarClassName: "bg-green-900/50",
    iconClassName: "text-green-400",
    icon: Bot,
  },
  tool: {
    avatarClassName: "bg-amber-900/50",
    iconClassName: "text-amber-400",
    icon: Wrench,
  },
  system: {
    avatarClassName: "bg-purple-900/50",
    iconClassName: "text-purple-400",
    contentClassName: "text-purple-300 italic",
    icon: Clock,
  },
};

function ToolCallList({ toolCalls }: { toolCalls: ToolCallPayload[] }) {
  if (toolCalls.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 space-y-1">
      {toolCalls.map((toolCall, index) => {
        const fn = toolCall.function;

        return (
          <div
            key={index}
            className="text-xs bg-zinc-800/50 rounded px-3 py-2 font-mono overflow-auto max-h-32"
          >
            <span className="text-amber-400">
              {fn ? fn.name : "tool_call"}
            </span>
            <span className="text-zinc-500 ml-2">
              {fn ? fn.arguments : JSON.stringify(toolCall)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function ChatMessageItem({ message }: { message: ChatMessageLike }) {
  const roleStyle = ROLE_STYLES[message.role];
  const Icon = roleStyle.icon;

  return (
    <div className="flex gap-3">
      <div className="shrink-0 mt-1">
        <div
          className={`w-7 h-7 rounded-full flex items-center justify-center ${roleStyle.avatarClassName}`}
        >
          <Icon className={`w-3.5 h-3.5 ${roleStyle.iconClassName}`} />
        </div>
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium text-zinc-400 uppercase">
            {message.role}
            {message.name && ` — ${message.name}`}
          </span>
          {message.isStreaming && (
            <Loader2 className="w-3 h-3 text-zinc-500 animate-spin" />
          )}
        </div>

        {message.reasoning_content && (
          <div className="mb-2 px-3 py-2 rounded bg-zinc-800/50 text-xs text-zinc-400 whitespace-pre-wrap max-h-32 overflow-auto">
            <span className="text-zinc-500 font-medium">Thinking: </span>
            {message.reasoning_content}
          </div>
        )}

        {message.content && (
          <div
            className={`text-sm whitespace-pre-wrap break-words ${roleStyle.contentClassName || ""}`}
          >
            {message.content}
          </div>
        )}

        {message.tool_calls && <ToolCallList toolCalls={message.tool_calls} />}
      </div>
    </div>
  );
}

export default ChatMessageItem;
