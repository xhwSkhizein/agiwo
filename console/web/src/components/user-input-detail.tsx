"use client";

import { UserInput } from "@/lib/api";

interface UserInputDetailProps {
  input: UserInput;
  showContext?: boolean;
  maxTextLength?: number;
}

interface ContentPart {
  type: string;
  text?: string;
  url?: string;
  mime_type?: string;
  detail?: string;
  metadata?: Record<string, unknown>;
}

interface ChannelContext {
  source: string;
  metadata?: Record<string, unknown>;
}

/**
 * Extract content parts from UserInput
 * Handles multiple formats:
 * - UserMessage with __type
 * - UserMessage without __type (Pydantic serialized)
 * - Simple content array
 */
function extractContentParts(input: UserInput): ContentPart[] | null {
  if (input === null || input === undefined) {
    return null;
  }

  if (typeof input !== "object") {
    return null;
  }

  const typed = input as Record<string, unknown>;

  // UserMessage format (with __type)
  if (typed.__type === "user_message") {
    const content = typed.content as ContentPart[] | undefined;
    return content || null;
  }

  // UserMessage format (without __type, Pydantic serialized)
  if (Array.isArray(typed.content)) {
    return typed.content as ContentPart[];
  }

  return null;
}

/**
 * Extract ChannelContext from UserInput
 */
function extractChannelContext(input: UserInput): ChannelContext | null {
  if (input === null || input === undefined) {
    return null;
  }

  if (typeof input !== "object") {
    return null;
  }

  const typed = input as Record<string, unknown>;

  // UserMessage format (with __type)
  if (typed.__type === "user_message") {
    const context = typed.context as ChannelContext | undefined;
    return context || null;
  }

  // UserMessage format (without __type, Pydantic serialized)
  if (typed.context && typeof typed.context === "object") {
    return typed.context as ChannelContext;
  }

  return null;
}

/**
 * Format text content with optional truncation
 */
function formatText(text: string, maxLength?: number): string {
  if (!maxLength || text.length <= maxLength) {
    return text;
  }
  return text.slice(0, maxLength) + "...";
}

/**
 * Get a short label for the content type
 */
function getContentTypeLabel(part: ContentPart): string {
  switch (part.type) {
    case "text":
      return "文本";
    case "image":
      return "图片";
    case "image_url":
      return "图片URL";
    case "file":
      return "文件";
    case "audio":
      return "音频";
    case "video":
      return "视频";
    default:
      return part.type;
  }
}

/**
 * Render a single content part
 */
function ContentPartView({
  part,
  maxTextLength,
}: {
  part: ContentPart;
  maxTextLength?: number;
}) {
  switch (part.type) {
    case "text":
      return (
        <p className="text-sm text-zinc-300 whitespace-pre-wrap">
          {formatText(part.text || "", maxTextLength)}
        </p>
      );

    case "image":
    case "image_url":
      return (
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">{getContentTypeLabel(part)}</span>
          {part.url && (
            <div className="text-xs text-zinc-400 break-all">{part.url}</div>
          )}
        </div>
      );

    case "file":
      return (
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">{getContentTypeLabel(part)}</span>
          {part.url && (
            <div className="text-xs text-zinc-400 break-all">{part.url}</div>
          )}
          {part.mime_type && (
            <div className="text-xs text-zinc-500">{part.mime_type}</div>
          )}
        </div>
      );

    default:
      return (
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">{getContentTypeLabel(part)}</span>
          <pre className="text-xs text-zinc-400 overflow-x-auto">
            {JSON.stringify(part, null, 2)}
          </pre>
        </div>
      );
  }
}

/**
 * Render ChannelContext metadata
 */
function ChannelContextView({ context }: { context: ChannelContext }) {
  const { source, metadata } = context;

  return (
    <div className="mt-2 pt-2 border-t border-zinc-800">
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <span>来源:</span>
        <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">{source}</span>
      </div>
      {metadata && Object.keys(metadata).length > 0 && (
        <div className="mt-1 space-y-0.5">
          {Object.entries(metadata).map(([key, value]) => (
            <div key={key} className="flex items-start gap-2 text-xs">
              <span className="text-zinc-500 min-w-[80px]">{key}:</span>
              <span className="text-zinc-400 break-all">
                {typeof value === "string" ? value : JSON.stringify(value)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Comprehensive UserInput display component.
 *
 * Features:
 * - Displays text, image, file, and other content types
 * - Shows ChannelContext (source and metadata)
 * - Handles multiple UserInput formats
 * - Optional text truncation
 */
export function UserInputDetail({
  input,
  showContext = true,
  maxTextLength = 500,
}: UserInputDetailProps) {
  // Handle simple string input
  if (typeof input === "string") {
    return (
      <p className="text-sm text-zinc-300 whitespace-pre-wrap">
        {formatText(input, maxTextLength)}
      </p>
    );
  }

  // Handle invalid input
  if (input === null || input === undefined) {
    return <span className="text-sm text-zinc-500">-</span>;
  }

  const contentParts = extractContentParts(input);
  const channelContext = extractChannelContext(input);

  // Handle unknown object format
  if (!contentParts) {
    return (
      <div className="space-y-2">
        <pre className="text-xs text-zinc-400 overflow-x-auto bg-zinc-900 p-2 rounded">
          {JSON.stringify(input, null, 2)}
        </pre>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Content Parts */}
      <div className="space-y-2">
        {contentParts.map((part, index) => (
          <div
            key={index}
            className="p-2 rounded bg-zinc-900/50 border border-zinc-800/50"
          >
            <ContentPartView part={part} maxTextLength={maxTextLength} />
          </div>
        ))}
      </div>

      {/* Channel Context */}
      {showContext && channelContext && (
        <ChannelContextView context={channelContext} />
      )}
    </div>
  );
}

/**
 * Compact UserInput display (inline, for tables and lists).
 * Shows first text snippet only, truncated.
 */
export function UserInputCompact({
  input,
  maxLength = 100,
}: {
  input: UserInput;
  maxLength?: number;
}) {
  if (typeof input === "string") {
    return (
      <span className="text-sm text-zinc-300 truncate">
        {formatText(input, maxLength)}
      </span>
    );
  }

  if (input === null || input === undefined) {
    return <span className="text-sm text-zinc-500">-</span>;
  }

  const contentParts = extractContentParts(input);

  if (!contentParts) {
    return (
      <span className="text-sm text-zinc-400 italic">复杂输入</span>
    );
  }

  // Find first text part
  const firstText = contentParts.find((p) => p.type === "text" && p.text)?.text;

  if (firstText) {
    return (
      <span className="text-sm text-zinc-300 truncate">
        {formatText(firstText, maxLength)}
      </span>
    );
  }

  // No text found, show content type summary
  const types = contentParts.map((p) => getContentTypeLabel(p)).join(", ");
  return <span className="text-sm text-zinc-400 italic">[{types}]</span>;
}

export default UserInputDetail;
