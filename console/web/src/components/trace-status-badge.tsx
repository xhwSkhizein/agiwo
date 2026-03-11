"use client";

import { PillBadge } from "@/components/pill-badge";

const STATUS_STYLES: Record<string, string> = {
  ok: "bg-green-900/50 text-green-400",
  error: "bg-red-900/50 text-red-400",
  running: "bg-blue-900/50 text-blue-400",
};

export function TraceStatusBadge({ status }: { status: string }) {
  const cls = STATUS_STYLES[status] || "bg-zinc-800 text-zinc-400";

  return (
    <PillBadge className={`text-xs px-1.5 py-0.5 rounded ${cls}`}>
      {status}
    </PillBadge>
  );
}

export default TraceStatusBadge;
