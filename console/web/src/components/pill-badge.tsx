"use client";

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

const pillBadgeVariants = {
  default: "bg-panel-strong text-ink-soft border-line",
  success: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  error: "bg-red-500/15 text-red-400 border-red-500/30",
  warning: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  info: "bg-blue-500/15 text-blue-400 border-blue-500/30",
  running: "bg-cyan-500/15 text-cyan-400 border-cyan-500/30",
  pending: "bg-panel-muted text-ink-muted border-line-strong",
};

type PillBadgeVariant = keyof typeof pillBadgeVariants;

type PillBadgeProps = {
  children: ReactNode;
  className?: string;
  variant?: PillBadgeVariant;
  dot?: boolean;
};

/**
 * Renders a pill-shaped badge with variant-specific styling and an optional status dot.
 *
 * @param children - Content displayed inside the badge
 * @param className - Additional class names appended to the badge container
 * @param variant - Visual style variant to apply (e.g., "default", "success", "error", "running")
 * @param dot - When `true`, renders a small colored dot indicating status
 * @returns The rendered badge element
 */
export function PillBadge({
  children,
  className,
  variant = "default",
  dot = false,
}: PillBadgeProps) {
  const variantClasses = pillBadgeVariants[variant];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full border font-medium",
        "transition-colors duration-150",
        variantClasses,
        className
      )}
    >
      {dot && (
        <span
          className={cn(
            "w-1.5 h-1.5 rounded-full",
            variant === "running" && "bg-cyan-400 animate-pulse",
            variant === "success" && "bg-emerald-400",
            variant === "error" && "bg-red-400",
            variant === "warning" && "bg-amber-400",
            variant === "info" && "bg-blue-400",
            variant === "pending" && "bg-zinc-400",
            variant === "default" && "bg-zinc-400"
          )}
        />
      )}
      {children}
    </span>
  );
}

export default PillBadge;
