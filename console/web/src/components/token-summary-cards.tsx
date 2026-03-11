"use client";

import type { ReactNode } from "react";

import { MetricCard } from "@/components/metric-card";
import {
  formatTokenCount,
  formatUsd,
} from "@/lib/metrics";

type TokenSummaryCardsProps = {
  cost: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens?: number | null;
  cacheReadTokens?: number | null;
  cacheCreationTokens?: number | null;
  className?: string;
  cardClassName?: string;
  labelClassName?: string;
  valueClassName?: string;
  costLabel?: string;
  inputOutputLabel?: string;
  totalLabel?: string;
  cacheLabel?: string;
  showTotal?: boolean;
  showCache?: boolean;
  extraCards?: ReactNode;
  extraCardsPosition?: "before" | "after";
};

export function TokenSummaryCards({
  cost,
  inputTokens,
  outputTokens,
  totalTokens = null,
  cacheReadTokens = null,
  cacheCreationTokens = null,
  className = "grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4",
  cardClassName = "",
  labelClassName = "",
  valueClassName = "text-lg font-medium",
  costLabel = "Run Cost",
  inputOutputLabel = "Input / Output",
  totalLabel = "Total Tokens",
  cacheLabel = "Cache R/C",
  showTotal = true,
  showCache = true,
  extraCards = null,
  extraCardsPosition = "before",
}: TokenSummaryCardsProps) {
  return (
    <div className={className}>
      {extraCardsPosition === "before" ? extraCards : null}
      <MetricCard
        label={costLabel}
        className={cardClassName}
        labelClassName={labelClassName}
        valueClassName={valueClassName}
        value={formatUsd(cost)}
      />
      <MetricCard
        label={inputOutputLabel}
        className={cardClassName}
        labelClassName={labelClassName}
        valueClassName={valueClassName}
        value={`${formatTokenCount(inputTokens)} / ${formatTokenCount(outputTokens)}`}
      />
      {showTotal && (
        <MetricCard
          label={totalLabel}
          className={cardClassName}
          labelClassName={labelClassName}
          valueClassName={valueClassName}
          value={formatTokenCount(totalTokens ?? 0)}
        />
      )}
      {showCache && (
        <MetricCard
          label={cacheLabel}
          className={cardClassName}
          labelClassName={labelClassName}
          valueClassName={valueClassName}
          value={
            `${formatTokenCount(cacheReadTokens ?? 0)} / `
            + `${formatTokenCount(cacheCreationTokens ?? 0)}`
          }
        />
      )}
      {extraCardsPosition === "after" ? extraCards : null}
    </div>
  );
}

export default TokenSummaryCards;
