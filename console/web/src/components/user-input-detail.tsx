"use client";

import type {
  ChannelContextPayload,
  ContentParts,
  ContentPartPayload,
  UserInput,
  UserMessage,
} from "@/lib/api";
import { PillBadge } from "@/components/pill-badge";

interface UserInputDetailProps {
  input: UserInput | null | undefined;
  showContext?: boolean;
  maxTextLength?: number;
}

interface UserInputCompactProps {
  input: UserInput | null | undefined;
  maxLength?: number;
  showContextBadge?: boolean;
  showMetadata?: boolean;
  showAttachmentBadge?: boolean;
}

type UserInputObject = UserMessage | ContentParts;
type LooseSerializedUserInput = {
  __type?: string;
  content?: ContentPartPayload[];
  parts?: ContentPartPayload[];
  context?: ChannelContextPayload | null;
};

/**
 * Extract content parts from UserInput
 * Handles multiple formats:
 * - UserMessage with __type
 * - ContentParts with __type
 * - UserMessage without __type (Pydantic serialized)
 * - Simple content/parts arrays
 */
function extractContentParts(input: UserInput): ContentPartPayload[] | null {
  if (input === null || input === undefined) {
    return null;
  }

  if (Array.isArray(input)) {
    return input;
  }

  if (typeof input !== "object") {
    return null;
  }

  const typed = input as UserInputObject;
  const looseTyped = input as LooseSerializedUserInput;

  // UserMessage format (with __type)
  if (typed.__type === "user_message") {
    const content = typed.content as ContentPartPayload[] | undefined;
    return content || null;
  }

  // ContentParts format (with __type)
  if (typed.__type === "content_parts") {
    const parts = typed.parts as ContentPartPayload[] | undefined;
    return parts || null;
  }

  // UserMessage format (without __type, Pydantic serialized)
  if (Array.isArray(looseTyped.content)) {
    return looseTyped.content;
  }

  // ContentParts format (without __type)
  if (Array.isArray(looseTyped.parts)) {
    return looseTyped.parts;
  }

  return null;
}

/**
 * Extract ChannelContext from UserInput
 */
function extractChannelContext(input: UserInput): ChannelContextPayload | null {
  if (input === null || input === undefined) {
    return null;
  }

  if (Array.isArray(input)) {
    return null;
  }

  if (typeof input !== "object") {
    return null;
  }

  const typed = input as UserInputObject;
  const looseTyped = input as LooseSerializedUserInput;

  // UserMessage format (with __type)
  if (typed.__type === "user_message") {
    const context = typed.context as ChannelContextPayload | undefined;
    return context || null;
  }

  // UserMessage format (without __type, Pydantic serialized)
  if (looseTyped.context && typeof looseTyped.context === "object") {
    return looseTyped.context;
  }

  return null;
}

function formatMetadataValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  const serialized = JSON.stringify(value);
  return serialized ?? String(value);
}

function getInputTypeLabel(input: UserInput): string | null {
  if (input === null || input === undefined) {
    return null;
  }

  if (Array.isArray(input)) {
    return "content_parts";
  }

  if (typeof input !== "object") {
    return null;
  }

  const typed = input as UserInputObject;
  return typeof typed.__type === "string" ? typed.__type : null;
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
function getContentTypeLabel(part: ContentPartPayload): string {
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
    case "input_audio":
      return "音频";
    case "video":
    case "video_url":
      return "视频";
    default:
      return part.type;
  }
}

function ResourceMeta({ part }: { part: ContentPartPayload }) {
  const entries = Object.entries(part.metadata || {});

  return (
    <>
      {part.mime_type && (
        <div className="text-xs text-zinc-500">{part.mime_type}</div>
      )}
      {part.detail && (
        <div className="text-xs text-zinc-500">{part.detail}</div>
      )}
      {entries.length > 0 && (
        <div className="space-y-0.5">
          {entries.map(([key, value]) => (
            <div key={key} className="text-xs text-zinc-500 break-all">
              {key}: {formatMetadataValue(value)}
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function ResourceView({ part }: { part: ContentPartPayload }) {
  return (
    <div className="space-y-1">
      <span className="text-xs text-zinc-500">{getContentTypeLabel(part)}</span>
      {part.url && (
        <a
          href={part.url}
          target="_blank"
          rel="noopener noreferrer"
          className="block text-xs text-blue-400 break-all hover:underline"
        >
          {part.url}
        </a>
      )}
      <ResourceMeta part={part} />
    </div>
  );
}

/**
 * Render a single content part
 */
function ContentPartView({
  part,
  maxTextLength,
}: {
  part: ContentPartPayload;
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
    case "audio":
    case "input_audio":
    case "video":
    case "video_url":
    case "file":
      return <ResourceView part={part} />;

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
function ChannelContextView({ context }: { context: ChannelContextPayload }) {
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
                {formatMetadataValue(value)}
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
  showContextBadge = false,
  showMetadata = false,
  showAttachmentBadge = false,
}: UserInputCompactProps) {
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
  const channelContext = extractChannelContext(input);
  const textParts = contentParts
    ?.filter((p) => p.type === "text" && p.text)
    .map((p) => p.text as string) || [];
  const textSummary = textParts.join(" ");
  const firstText = textParts[0];
  const hasAttachments = contentParts?.some((p) => p.type !== "text") || false;
  const metadataEntries = Object.entries(channelContext?.metadata || {});
  const inputTypeLabel = getInputTypeLabel(input);
  const contextBadge =
    channelContext?.source ||
    (showContextBadge && inputTypeLabel === "content_parts"
      ? inputTypeLabel
      : null);
  const shouldRenderDetailed =
    Boolean(contextBadge) ||
    (showAttachmentBadge && hasAttachments) ||
    (showMetadata && metadataEntries.length > 0);

  if (!contentParts) {
    return (
      <span className="text-sm text-zinc-400 italic">复杂输入</span>
    );
  }

  const summaryText =
    textSummary ||
    (hasAttachments
      ? `[${contentParts.map((p) => getContentTypeLabel(p)).join(", ")}]`
      : "(无文本内容)");

  if (shouldRenderDetailed) {
    return (
      <div className="space-y-1.5">
        <div className="flex items-center gap-2 min-w-0">
          {contextBadge && (
            <PillBadge className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700/50">
              {contextBadge}
            </PillBadge>
          )}
          <span className="min-w-0 flex-1 text-sm text-zinc-300 truncate">
            {formatText(summaryText, maxLength)}
          </span>
          {showAttachmentBadge && hasAttachments && (
            <span className="shrink-0 text-[10px] px-1 py-0.5 rounded bg-zinc-800 text-zinc-500">
              +附件
            </span>
          )}
        </div>
        {showMetadata && metadataEntries.length > 0 && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            {metadataEntries.map(([key, value]) => (
              <span key={key} className="text-[10px] text-zinc-500">
                <span className="text-zinc-600">{key}:</span>
                <span className="text-zinc-400 ml-1">
                  {formatMetadataValue(value)}
                </span>
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }

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
