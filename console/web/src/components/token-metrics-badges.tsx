"use client";

import type { ParsedTokenMetrics } from "@/lib/metrics";
import {
  formatDurationMs,
  formatTokenCount,
  formatUsageSource,
  formatUsd,
} from "@/lib/metrics";

type TokenMetricsBadgesProps = {
  metrics: ParsedTokenMetrics;
  modelName?: string | null;
  showCacheRead?: boolean;
  showCacheCreation?: boolean;
  showDuration?: boolean;
  showModelName?: boolean;
  showUsageSource?: boolean;
  className?: string;
  chipClassName?: string;
};

type BadgeItem = {
  key: string;
  content: string;
  textClassName: string;
  extraClassName?: string;
};

export function TokenMetricsBadges({
  metrics,
  modelName,
  showCacheRead = true,
  showCacheCreation = true,
  showDuration = false,
  showModelName = false,
  showUsageSource = true,
  className = "grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs",
  chipClassName = "bg-zinc-800/60",
}: TokenMetricsBadgesProps) {
  const items: BadgeItem[] = [
    {
      key: "cost",
      content: `cost ${formatUsd(metrics.tokenCost)}`,
      textClassName: "text-zinc-300",
    },
    {
      key: "input",
      content: `in ${formatTokenCount(metrics.inputTokens)}`,
      textClassName: "text-zinc-400",
    },
    {
      key: "output",
      content: `out ${formatTokenCount(metrics.outputTokens)}`,
      textClassName: "text-zinc-400",
    },
    {
      key: "total",
      content: `total ${formatTokenCount(metrics.totalTokens)}`,
      textClassName: "text-zinc-400",
    },
  ];

  if (showCacheRead) {
    items.push({
      key: "cache-read",
      content: `cache read ${formatTokenCount(metrics.cacheReadTokens)}`,
      textClassName: "text-zinc-500",
    });
  }

  if (showCacheCreation) {
    items.push({
      key: "cache-create",
      content: `cache create ${formatTokenCount(metrics.cacheCreationTokens)}`,
      textClassName: "text-zinc-500",
    });
  }

  if (showDuration) {
    items.push({
      key: "duration",
      content: formatDurationMs(metrics.durationMs),
      textClassName: "text-zinc-500",
    });
  }

  if (showUsageSource && metrics.usageSource) {
    items.push({
      key: "usage-source",
      content: formatUsageSource(metrics.usageSource),
      textClassName:
        metrics.usageSource === "provider"
          ? "text-emerald-400"
          : metrics.usageSource === "estimated"
          ? "text-amber-400"
          : "text-sky-400",
    });
  }

  if (showModelName && modelName) {
    items.push({
      key: "model-name",
      content: modelName,
      textClassName: "text-zinc-500",
      extraClassName: "truncate",
    });
  }

  return (
    <div className={className}>
      {items.map((item) => (
        <span
          key={item.key}
          className={`px-2 py-1 rounded ${chipClassName} ${item.textClassName} ${item.extraClassName || ""}`}
        >
          {item.content}
        </span>
      ))}
    </div>
  );
}

export default TokenMetricsBadges;
