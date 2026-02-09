"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { ArrowLeft, User, Bot, Wrench } from "lucide-react";
import Link from "next/link";
import { getSessionSteps } from "@/lib/api";
import type { StepResponse } from "@/lib/api";

function StepCard({ step }: { step: StepResponse }) {
  const isUser = step.role === "user";
  const isAssistant = step.role === "assistant";
  const isTool = step.role === "tool";

  const content =
    typeof step.content === "string"
      ? step.content
      : step.content
      ? JSON.stringify(step.content, null, 2)
      : "";

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

      {content && (
        <div className="text-sm whitespace-pre-wrap break-words max-h-96 overflow-auto">
          {content}
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
