"use client";

import { PillBadge } from "@/components/pill-badge";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-yellow-900/50 text-yellow-400",
  running: "bg-blue-900/50 text-blue-400",
  waiting: "bg-amber-900/50 text-amber-300",
  idle: "bg-purple-900/50 text-purple-400",
  queued: "bg-cyan-900/50 text-cyan-300",
  completed: "bg-green-900/50 text-green-400",
  failed: "bg-red-900/50 text-red-400",
};

export function SchedulerStatusBadge({ status }: { status: string }) {
  const cls = STATUS_STYLES[status] || "bg-zinc-800 text-zinc-400";

  return (
    <PillBadge className={`text-xs px-1.5 py-0.5 rounded ${cls}`}>
      {status}
    </PillBadge>
  );
}

export default SchedulerStatusBadge;
