"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { ArrowLeft, User, Bot, Wrench, FileImage, FileAudio, FileVideo, FileText, Paperclip } from "lucide-react";
import Link from "next/link";
import { getSessionSteps } from "@/lib/api";
import type { StepResponse, UserMessage, UserInput } from "@/lib/api";

function ContentPartItem({ part }: { part: { type: string; text?: string; url?: string; mime_type?: string } }) {
  const { type, text, url, mime_type } = part;

  // Text content
  if (type === "text" && text) {
    return (
      <div className="text-sm text-zinc-200 whitespace-pre-wrap break-words">
        {text}
      </div>
    );
  }

  // Image
  if (type === "image" || type === "image_url" || mime_type?.startsWith("image/")) {
    return (
      <div className="flex items-start gap-2 p-2 rounded bg-zinc-800/50">
        <FileImage className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
        <div className="min-w-0">
          <span className="text-xs text-zinc-400">[图片]</span>
          {url && (
            <a href={url} target="_blank" rel="noopener noreferrer" className="block text-xs text-blue-400 truncate hover:underline">
              {url}
            </a>
          )}
          {mime_type && <span className="text-xs text-zinc-600 ml-2">{mime_type}</span>}
        </div>
      </div>
    );
  }

  // Audio
  if (type === "audio" || type === "input_audio" || mime_type?.startsWith("audio/")) {
    return (
      <div className="flex items-start gap-2 p-2 rounded bg-zinc-800/50">
        <FileAudio className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="min-w-0">
          <span className="text-xs text-zinc-400">[音频]</span>
          {url && <span className="block text-xs text-zinc-500 truncate">{url}</span>}
          {mime_type && <span className="text-xs text-zinc-600 ml-2">{mime_type}</span>}
        </div>
      </div>
    );
  }

  // Video
  if (type === "video" || type === "video_url" || mime_type?.startsWith("video/")) {
    return (
      <div className="flex items-start gap-2 p-2 rounded bg-zinc-800/50">
        <FileVideo className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
        <div className="min-w-0">
          <span className="text-xs text-zinc-400">[视频]</span>
          {url && <span className="block text-xs text-zinc-500 truncate">{url}</span>}
          {mime_type && <span className="text-xs text-zinc-600 ml-2">{mime_type}</span>}
        </div>
      </div>
    );
  }

  // File
  if (type === "file" || url) {
    return (
      <div className="flex items-start gap-2 p-2 rounded bg-zinc-800/50">
        <Paperclip className="w-4 h-4 text-zinc-400 shrink-0 mt-0.5" />
        <div className="min-w-0">
          <span className="text-xs text-zinc-400">[文件]</span>
          {url && (
            <a href={url} target="_blank" rel="noopener noreferrer" className="block text-xs text-blue-400 truncate hover:underline">
              {url}
            </a>
          )}
          {mime_type && <span className="text-xs text-zinc-600 ml-2">{mime_type}</span>}
        </div>
      </div>
    );
  }

  // Unknown type
  return (
    <div className="flex items-start gap-2 p-2 rounded bg-zinc-800/50">
      <FileText className="w-4 h-4 text-zinc-500 shrink-0 mt-0.5" />
      <div className="min-w-0 text-xs text-zinc-500">
        <span className="text-zinc-400">[{type}]</span>
        {text && <span className="ml-2">{text}</span>}
        {url && <span className="block truncate">{url}</span>}
      </div>
    </div>
  );
}

function UserMessageContent({ content }: { content: unknown }) {
  // Try to parse as UserMessage
  if (typeof content !== "object" || content === null) {
    return <div className="text-sm text-zinc-200">{typeof content === "string" ? content : String(content)}</div>;
  }

  const typed = content as Record<string, unknown>;
  const type = typed.__type;

  // UserMessage format
  if (type === "user_message") {
    const userMsg = content as UserMessage;
    const { content: parts, context } = userMsg;

    return (
      <div className="space-y-3">
        {/* Content parts */}
        {parts && parts.length > 0 && (
          <div className="space-y-2">
            {parts.map((part, i) => (
              <ContentPartItem key={i} part={part} />
            ))}
          </div>
        )}

        {/* Context info */}
        {context && (
          <div className="pt-2 border-t border-zinc-800/50 space-y-2">
            {/* Source */}
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Source</span>
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-900/30 text-emerald-400 border border-emerald-800/30">
                {context.source}
              </span>
            </div>

            {/* Metadata */}
            {context.metadata && Object.keys(context.metadata).length > 0 && (
              <div className="space-y-1">
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Metadata</span>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  {Object.entries(context.metadata).map(([key, value]) => (
                    <div key={key} className="flex items-center gap-2 min-w-0">
                      <span className="text-zinc-500 shrink-0">{key}:</span>
                      <span className="text-zinc-300 truncate font-mono">
                        {typeof value === "string" ? value : JSON.stringify(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // ContentParts format
  if (type === "content_parts") {
    const parts = typed.parts as Array<{ type: string; text?: string; url?: string; mime_type?: string }> | undefined;
    if (parts && parts.length > 0) {
      return (
        <div className="space-y-2">
          {parts.map((part, i) => (
            <ContentPartItem key={i} part={part} />
          ))}
        </div>
      );
    }
  }

  // Fallback to string representation
  return <div className="text-sm text-zinc-200">{JSON.stringify(content, null, 2)}</div>;
}

function StepCard({ step }: { step: StepResponse }) {
  const isUser = step.role === "user";
  const isAssistant = step.role === "assistant";
  const isTool = step.role === "tool";

  // For user steps, use the full UserMessageContent component
  // For other steps, use simple text display
  const content = step.content;

  return (
    <div
      className={`rounded-lg border p-4 ${
        isUser
          ? "border-blue-800/50 bg-blue-950/20"
          : isTool
          ? "border-amber-800/50 bg-amber-950/20"
          : "border-zinc-800 bg-zinc-900"
      }`}
    >
      <div className="flex items-center gap-2 mb-2">
        {isUser && <User className="w-4 h-4 text-blue-400" />}
        {isAssistant && <Bot className="w-4 h-4 text-green-400" />}
        {isTool && <Wrench className="w-4 h-4 text-amber-400" />}
        <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
          {step.role}
          {isTool && step.name && ` — ${step.name}`}
        </span>
        <span className="text-xs text-zinc-600 ml-auto">#{step.sequence}</span>
      </div>

      {step.reasoning_content && (
        <div className="mb-2 px-3 py-2 rounded bg-zinc-800/50 text-xs text-zinc-400 whitespace-pre-wrap max-h-48 overflow-auto">
          <span className="text-zinc-500 font-medium">Thinking: </span>
          {step.reasoning_content}
        </div>
      )}

      {Boolean(content) && (
        <div className="max-h-96 overflow-auto">
          {isUser ? (
            <UserMessageContent content={content} />
          ) : (
            <div className="text-sm whitespace-pre-wrap break-words">
              {typeof content === "string"
                ? content
                : JSON.stringify(content, null, 2)}
            </div>
          )}
        </div>
      )}

      {step.tool_calls && step.tool_calls.length > 0 && (
        <div className="mt-2 space-y-1">
          {step.tool_calls.map((tc, i) => (
            <div
              key={i}
              className="text-xs bg-zinc-800/50 rounded px-3 py-2 font-mono overflow-auto max-h-48"
            >
              <span className="text-amber-400">
                {(tc as Record<string, unknown>).function
                  ? ((tc as Record<string, unknown>).function as Record<string, unknown>).name as string
                  : "tool_call"}
              </span>
              <span className="text-zinc-500 ml-2">
                {(tc as Record<string, unknown>).function
                  ? ((tc as Record<string, unknown>).function as Record<string, unknown>).arguments as string
                  : JSON.stringify(tc)}
              </span>
            </div>
          ))}
        </div>
      )}

      {step.metrics && (
        <div className="mt-2 flex gap-3 text-xs text-zinc-500">
          {(step.metrics as Record<string, unknown>).total_tokens != null && (
            <span>{String((step.metrics as Record<string, unknown>).total_tokens)} tokens</span>
          )}
          {(step.metrics as Record<string, unknown>).duration_ms != null && (
            <span>{Math.round(Number((step.metrics as Record<string, unknown>).duration_ms))}ms</span>
          )}
          {(step.metrics as Record<string, unknown>).model_name != null && (
            <span>{String((step.metrics as Record<string, unknown>).model_name)}</span>
          )}
        </div>
      )}
    </div>
  );
}

export default function SessionDetailPage() {
  const params = useParams();
  const sessionId = params.id as string;
  const [steps, setSteps] = useState<StepResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSessionSteps(sessionId)
      .then(setSteps)
      .catch(() => setSteps([]))
      .finally(() => setLoading(false));
  }, [sessionId]);

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Link
          href="/sessions"
          className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <h1 className="text-xl font-semibold">Session Detail</h1>
          <p className="text-xs text-zinc-500 font-mono mt-0.5">{sessionId}</p>
        </div>
      </div>

      {loading ? (
        <div className="text-zinc-500">Loading steps...</div>
      ) : steps.length === 0 ? (
        <div className="text-zinc-500 text-center py-12">No steps found</div>
      ) : (
        <div className="space-y-3">
          {steps.map((step) => (
            <StepCard key={step.id} step={step} />
          ))}
        </div>
      )}
    </div>
  );
}
