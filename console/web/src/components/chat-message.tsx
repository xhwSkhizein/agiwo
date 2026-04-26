"use client";

import { useState } from "react";
import {
  Bot,
  ChevronDown,
  ChevronRight,
  Clock,
  Loader2,
  User,
  Wrench,
} from "lucide-react";
import type { ChatMessage, ChatRole } from "@/lib/chat-types";
import type { ToolCallPayload } from "@/lib/api";
import { JsonDisclosure } from "@/components/json-disclosure";
import { UserInputDetail } from "@/components/user-input-detail";

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
    <div className="mt-2 flex flex-wrap gap-1.5">
      {toolCalls.map((toolCall, index) => {
        const fn = toolCall.function;
        const args = fn ? fn.arguments : JSON.stringify(toolCall);

        return (
          <details
            key={index}
            className="group rounded-md border border-line bg-panel-muted text-xs"
          >
            <summary className="flex cursor-pointer list-none items-center gap-1.5 px-2.5 py-1.5 text-ink-muted transition-colors hover:text-foreground">
              <Wrench className="h-3 w-3 text-warning" />
              <span className="font-medium">{fn ? fn.name : "tool_call"}</span>
              <span className="max-w-56 truncate font-mono text-[11px] text-ink-faint">
                {args}
              </span>
            </summary>
            <pre className="max-h-40 overflow-auto border-t border-line px-2.5 py-2 font-mono text-[11px] text-ink-muted whitespace-pre-wrap break-words">
              {args}
            </pre>
          </details>
        );
      })}
    </div>
  );
}

/**
 * Render a collapsible JSON disclosure for non-null, non-string payloads.
 *
 * @param value - The payload to display; ignored when `null`, `undefined`, or a `string`.
 * @returns A React element showing a labeled JSON disclosure of `value`, or `null` when the value is `null`, `undefined`, or a `string`.
 */
function RawPayload({ value }: { value: unknown }) {
  if (value === null || value === undefined || typeof value === "string") {
    return null;
  }

  return <JsonDisclosure className="mt-2" label="Raw payload" value={value} />;
}

function OriginalContentToggle({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
      >
        {expanded ? (
          <ChevronDown className="w-3 h-3" />
        ) : (
          <ChevronRight className="w-3 h-3" />
        )}
        {expanded ? "Hide original result" : "View original result"}
      </button>
      {expanded && (
        <div className="mt-1 px-3 py-2 rounded bg-zinc-800/50 text-xs text-zinc-400 whitespace-pre-wrap max-h-64 overflow-auto">
          {content}
        </div>
      )}
    </div>
  );
}

function compactText(value: string | undefined, maxLength: number): string {
  if (!value) {
    return "No text result";
  }
  const trimmed = value.trim();
  if (trimmed.length <= maxLength) {
    return trimmed;
  }
  return `${trimmed.slice(0, maxLength - 1)}…`;
}

function ToolResultMessage({ message }: { message: ChatMessage }) {
  const preview = compactText(message.text, 320);
  const hasLongResult = Boolean(message.text && message.text.trim().length > 320);

  return (
    <details className="group rounded-lg border border-line bg-panel-muted/70">
      <summary className="cursor-pointer list-none px-3 py-2">
        <div className="flex min-w-0 items-start gap-2">
          <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-warning/10">
            <Wrench className="h-3 w-3 text-warning" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium text-ink-soft">
                {message.name || "tool result"}
              </span>
              {message.sequence !== undefined && (
                <span className="text-[11px] text-ink-faint">seq {message.sequence}</span>
              )}
              {message.sourceAgentId && (
                <span className="font-mono text-[11px] text-ink-faint">
                  {message.sourceAgentId.slice(0, 8)}
                </span>
              )}
              <span className="ml-auto text-[11px] text-ink-faint group-open:hidden">
                expand
              </span>
            </div>
            <p className="mt-1 max-h-16 overflow-hidden text-xs leading-5 text-ink-muted">
              {preview}
            </p>
          </div>
        </div>
      </summary>
      <div className="border-t border-line px-3 pb-3 pt-2">
        {message.text && (
          <div className="max-h-72 overflow-auto rounded-md bg-panel px-3 py-2 text-xs leading-5 text-ink-soft whitespace-pre-wrap break-words">
            {message.text}
          </div>
        )}
        {hasLongResult && (
          <p className="mt-2 text-[11px] text-ink-faint">
            Long result is collapsed by default to keep the run narrative readable.
          </p>
        )}
        {message.originalContent && (
          <OriginalContentToggle content={message.originalContent} />
        )}
        <RawPayload value={message.rawContent} />
      </div>
    </details>
  );
}

/**
 * Render a single chat message with a role-specific avatar, header, and conditional content sections.
 *
 * The rendered content may include a streaming indicator, a "Thinking" reasoning block, a user input detail,
 * the message text (when no `userInput` is present), a list of tool calls, and a raw payload disclosure depending
 * on which fields are present on `message`.
 *
 * @param message - The chat message to render. Relevant fields: `role`, `name`, `sourceAgentId`, `isStreaming`,
 *   `reasoningContent`, `userInput`, `text`, `toolCalls`, and `rawContent`.
 * @returns A JSX element representing the formatted chat message.
 */
export function ChatMessageItem({ message }: { message: ChatMessage }) {
  if (message.role === "tool") {
    return (
      <div className="pl-10">
        <ToolResultMessage message={message} />
      </div>
    );
  }

  const roleStyle = ROLE_STYLES[message.role];
  const Icon = roleStyle.icon;

  return (
    <div className="flex gap-3">
      <div className="shrink-0 mt-1">
        <div
          className={`w-7 h-7 rounded-lg flex items-center justify-center ${roleStyle.avatarClassName}`}
        >
          <Icon className={`w-3.5 h-3.5 ${roleStyle.iconClassName}`} />
        </div>
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium text-ink-muted uppercase">
            {message.role}
            {message.name && ` — ${message.name}`}
            {message.sourceAgentId && ` — ${message.sourceAgentId.slice(0, 8)}`}
          </span>
          {message.isStreaming && (
            <Loader2 className="w-3 h-3 text-zinc-500 animate-spin" />
          )}
        </div>

        {message.reasoningContent && (
          <div className="mb-2 max-h-32 overflow-auto rounded-md border border-line bg-panel-muted px-3 py-2 text-xs text-ink-muted whitespace-pre-wrap">
            <span className="text-ink-faint font-medium">Thinking: </span>
            {message.reasoningContent}
          </div>
        )}

        {message.userInput && (
          <div className="max-h-96 overflow-auto">
            <UserInputDetail input={message.userInput} maxTextLength={2000} />
          </div>
        )}

        {!message.userInput && message.text && (
          <div
            className={`text-sm leading-6 whitespace-pre-wrap break-words ${roleStyle.contentClassName || ""}`}
          >
            {message.text}
          </div>
        )}

        {message.originalContent && (
          <OriginalContentToggle content={message.originalContent} />
        )}

        {message.toolCalls && <ToolCallList toolCalls={message.toolCalls} />}
        <RawPayload value={message.rawContent} />
      </div>
    </div>
  );
}

export default ChatMessageItem;
