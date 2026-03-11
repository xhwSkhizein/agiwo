"use client";

import type { ReactNode } from "react";

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
  className = "",
  labelClassName = "",
  valueClassName = "",
}: MetricCardProps) {
  return (
    <div className={`rounded-lg border border-zinc-800 bg-zinc-900 p-4 ${className}`}>
      <p className={`text-xs text-zinc-500 ${labelClassName}`}>{label}</p>
      <div className={`mt-1 ${valueClassName}`}>{value}</div>
    </div>
  );
}

export default MetricCard;
