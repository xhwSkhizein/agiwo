"use client";

import { PillBadge } from "@/components/pill-badge";

const STATUS_VARIANTS: Record<string, { variant: Parameters<typeof PillBadge>[0]["variant"]; dot?: boolean }> = {
  pending: { variant: "pending", dot: false },
  running: { variant: "running", dot: true },
  waiting: { variant: "warning", dot: false },
  idle: { variant: "info", dot: false },
  queued: { variant: "info", dot: true },
  completed: { variant: "success", dot: false },
  failed: { variant: "error", dot: false },
};

export function SchedulerStatusBadge({ status }: { status: string }) {
  const config = STATUS_VARIANTS[status] || { variant: "default", dot: false };

  return (
    <PillBadge variant={config.variant} dot={config.dot}>
      {status}
    </PillBadge>
  );
}

export default SchedulerStatusBadge;
