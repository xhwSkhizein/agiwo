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
