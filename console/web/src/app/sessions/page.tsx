"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listSessions } from "@/lib/api";
import type { SessionSummary, UserInput, UserMessage } from "@/lib/api";

function UserInputDetail({ input }: { input: UserInput | null }) {
  if (!input) return <span className="text-zinc-500 text-sm">—</span>;

  // Plain string
  if (typeof input === "string") {
    return <span className="text-sm text-zinc-200 truncate">{input}</span>;
  }

  // Object with __type
  if (typeof input === "object" && input !== null && "__type" in input) {
    const typed = input as Record<string, unknown>;
    const type = typed.__type;

    // UserMessage format
    if (type === "user_message") {
      const userMsg = input as UserMessage;
      const textParts = userMsg.content?.filter(p => p.type === "text" && p.text).map(p => p.text) || [];
      const hasAttachments = userMsg.content?.some(p => p.type !== "text" && (p.url || p.mime_type));

      return (
        <div className="space-y-1.5">
          {/* Main row: source + text + attachments */}
          <div className="flex items-center gap-2 min-w-0">
            {/* Source tag */}
            {userMsg.context && (
              <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700/50">
                {userMsg.context.source}
              </span>
            )}

            {/* Main text content */}
            <span className="text-sm text-zinc-200 truncate">
              {textParts.join(" ") || "(无文本内容)"}
            </span>

            {/* Attachment indicator */}
            {hasAttachments && (
              <span className="shrink-0 text-[10px] px-1 py-0.5 rounded bg-zinc-800 text-zinc-500">
                +附件
              </span>
            )}
          </div>

          {/* Metadata info - 直接展示 */}
          {userMsg.context?.metadata && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              {Object.entries(userMsg.context.metadata).map(([key, value]) => (
                <span key={key} className="text-[10px] text-zinc-500">
                  <span className="text-zinc-600">{key}:</span>
                  <span className="text-zinc-400 ml-1">
                    {typeof value === "string" ? value : JSON.stringify(value)}
                  </span>
                </span>
              ))}
            </div>
          )}
        </div>
      );
    }

    // ContentParts format
    if (type === "content_parts") {
      const parts = typed.parts as Array<{ type: string; text?: string; url?: string }> | undefined;
      const textParts = parts?.filter(p => p.type === "text" && p.text).map(p => p.text) || [];
      const hasAttachments = parts?.some(p => p.type !== "text");

      return (
        <div className="flex items-center gap-2 min-w-0">
          <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700/50">
            {typed.__type as string}
          </span>
          <span className="text-sm text-zinc-200 truncate">
            {textParts.join(" ") || "(无文本内容)"}
          </span>
          {hasAttachments && (
            <span className="shrink-0 text-[10px] px-1 py-0.5 rounded bg-zinc-800 text-zinc-500">
              +附件
            </span>
          )}
        </div>
      );
    }
  }

  // Fallback: JSON stringify
  return (
    <span className="text-xs text-zinc-400 font-mono truncate">
      {JSON.stringify(input).slice(0, 100)}
    </span>
  );
}

export default function SessionsPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listSessions(50)
      .then(setSessions)
      .catch(() => setSessions([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Sessions</h1>
        <p className="text-sm text-zinc-400 mt-1">All conversation sessions</p>
      </div>

      {loading ? (
        <div className="text-zinc-500">Loading...</div>
      ) : sessions.length === 0 ? (
        <div className="text-zinc-500 text-center py-12">No sessions found</div>
      ) : (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 divide-y divide-zinc-800">
          {sessions.map((s) => (
            <Link
              key={s.session_id}
              href={`/sessions/${s.session_id}`}
              className="block px-5 py-4 hover:bg-zinc-800/50 transition-colors"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="text-sm">
                    <UserInputDetail input={s.last_user_input} />
                  </div>
                  {s.last_response && (
                    <p className="text-xs text-zinc-500 mt-2 truncate">
                      {s.last_response}
                    </p>
                  )}
                </div>
                <div className="text-right shrink-0">
                  <p className="text-xs text-zinc-400">{s.run_count} runs</p>
                  <p className="text-xs text-zinc-600 mt-0.5">
                    {s.agent_id || "unknown"}
                  </p>
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
