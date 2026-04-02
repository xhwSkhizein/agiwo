"use client";

import type { ReactNode } from "react";

type PillBadgeProps = {
  children: ReactNode;
  className?: string;
};

export function PillBadge({
  children,
  className = "text-xs px-1.5 py-0.5 rounded whitespace-nowrap",
}: PillBadgeProps) {
  return <span className={className}>{children}</span>;
}

export default PillBadge;
