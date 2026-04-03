"use client";

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type MetricCardProps = {
  label: string;
  value: ReactNode;
  className?: string;
  labelClassName?: string;
  valueClassName?: string;
};

/**
 * Renders a styled metric card with a header label and a main value.
 *
 * The component applies a default rounded panel style and merges any provided
 * `className`, `labelClassName`, and `valueClassName` with those defaults to
 * allow visual overrides.
 *
 * @param label - Text displayed in the card header
 * @param value - Content rendered as the card's main value (any ReactNode)
 * @param className - Optional classes appended to the card container
 * @param labelClassName - Optional classes appended to the label element
 * @param valueClassName - Optional classes appended to the value element
 * @returns The rendered JSX element for the metric card
 */
export function MetricCard({
  label,
  value,
  className,
  labelClassName,
  valueClassName,
}: MetricCardProps) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-line bg-panel p-4 shadow-sm transition-all duration-200",
        "hover:border-line-strong hover:bg-panel-strong",
        className
      )}
    >
      <p
        className={cn(
          "text-[11px] font-medium uppercase tracking-[0.16em] text-ink-faint",
          labelClassName,
        )}
      >
        {label}
      </p>
      <div className={cn("mt-1.5 text-2xl font-semibold text-foreground", valueClassName)}>
        {value}
      </div>
    </div>
  );
}

export default MetricCard;
