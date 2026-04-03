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

/**
 * Render a PillBadge that represents a scheduler job status.
 *
 * The badge's visual variant and optional dot are chosen based on `status`; unknown statuses use a default appearance.
 *
 * @param status - The scheduler status label to display and style (e.g. "pending", "running", "waiting", "idle", "queued", "completed", "failed")
 * @returns A JSX element that renders a PillBadge showing `status`
 */
export function SchedulerStatusBadge({ status }: { status: string }) {
  const config = STATUS_VARIANTS[status] || { variant: "default", dot: false };

  return (
    <PillBadge variant={config.variant} dot={config.dot}>
      {status}
    </PillBadge>
  );
}

export default SchedulerStatusBadge;
