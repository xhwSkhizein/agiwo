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
  ok: { variant: "success", dot: false },
  error: { variant: "error", dot: false },
};

/**
 * Render a pill-style badge representing the given trace status.
 *
 * @param status - Status key that determines the badge's visual variant. Supported values: `pending`, `running`, `waiting`, `idle`, `queued`, `completed`, `failed`, `ok`, `error`. If an unknown status is provided, the badge uses the `"default"` variant with no dot.
 * @returns A JSX element rendering a PillBadge whose variant and optional dot reflect the provided status.
 */
export function TraceStatusBadge({ status }: { status: string }) {
  const config = STATUS_VARIANTS[status] || { variant: "default", dot: false };

  return (
    <PillBadge variant={config.variant} dot={config.dot}>
      {status}
    </PillBadge>
  );
}

export default TraceStatusBadge;
