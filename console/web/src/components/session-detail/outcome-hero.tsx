"use client";

import { PillBadge } from "@/components/pill-badge";
import { MonoText } from "@/components/mono-text";
import type { ObjectiveView } from "@/lib/api";
import {
  objectiveFocusText,
  objectiveStatusVariant,
} from "@/lib/objective-status";
import { cn } from "@/lib/utils";

type OutcomeHeroProps = {
  objective: ObjectiveView | null;
  sessionId: string;
  onRefresh?: () => void;
};

function heroTone(status: string | undefined): string {
  const normalized = (status ?? "").toUpperCase();
  if (normalized === "COMPLETED") {
    return "from-emerald-500/15 via-panel to-panel border-emerald-500/25";
  }
  if (normalized === "FAILED") {
    return "from-red-500/15 via-panel to-panel border-red-500/25";
  }
  if (normalized === "RUNNING" || normalized === "DRAINING") {
    return "from-cyan-500/15 via-panel to-panel border-cyan-500/25";
  }
  if (
    normalized === "WAITING_USER" ||
    normalized === "BUDGET_PAUSED" ||
    normalized === "USER_PAUSED"
  ) {
    return "from-amber-500/15 via-panel to-panel border-amber-500/25";
  }
  return "from-panel-muted via-panel to-panel border-line";
}

function kicker(status: string | undefined, hasDelivery: boolean): string {
  const normalized = (status ?? "").toUpperCase();
  if (hasDelivery || normalized === "COMPLETED") return "Delivered";
  if (normalized === "WAITING_USER") return "Needs your reply";
  if (normalized === "BUDGET_PAUSED") return "Budget paused";
  if (normalized === "USER_PAUSED") return "Paused";
  if (normalized === "DRAINING") return "Draining";
  if (normalized === "FAILED") return "Failed";
  if (normalized === "RUNNING") return "In progress";
  return "Objective";
}

export function OutcomeHero({
  objective,
  sessionId,
  onRefresh,
}: OutcomeHeroProps) {
  if (!objective) {
    return (
      <section className="rounded-xl border border-dashed border-line bg-panel/40 px-4 py-3">
        <p className="text-xs uppercase tracking-wide text-ink-faint">Outcome</p>
        <p className="mt-1 text-sm text-ink-muted">
          No Objective yet for this session. Runs and steps below still show
          execution history.
        </p>
        <p className="mt-2 font-mono text-xs text-ink-faint">{sessionId}</p>
      </section>
    );
  }

  const focus = objectiveFocusText(objective);
  const hasDelivery = Boolean(objective.delivery_report);

  return (
    <section
      className={cn(
        "rounded-xl border bg-gradient-to-br px-4 py-3.5",
        heroTone(objective.status),
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1.5">
          <p className="text-[11px] font-semibold uppercase tracking-[0.06em] text-ink-faint">
            {kicker(objective.status, hasDelivery)}
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <PillBadge variant={objectiveStatusVariant(objective.status)} dot>
              {objective.status}
            </PillBadge>
            <MonoText className="text-[11px] text-ink-faint">
              {objective.objective_id}
            </MonoText>
          </div>
          <p className="max-w-3xl text-sm leading-6 text-foreground line-clamp-3 whitespace-pre-wrap">
            {focus}
          </p>
        </div>
        {onRefresh ? (
          <button
            type="button"
            onClick={onRefresh}
            className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-muted hover:border-line-strong hover:text-foreground"
          >
            Refresh
          </button>
        ) : null}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 border-t border-line/70 pt-2.5 text-xs text-ink-muted">
        <span>
          Handoffs{" "}
          <b className="font-mono font-medium text-foreground">
            {objective.budget.handoffs.used}/{objective.budget.handoffs.limit}
          </b>
        </span>
        <span>
          LLM $
          <b className="font-mono font-medium text-foreground">
            {objective.budget.llm_cost_usd.used.toFixed(2)}/
            {objective.budget.llm_cost_usd.limit}
          </b>
        </span>
        {objective.delivery_outcome_id ? (
          <span>
            Outcome{" "}
            <MonoText className="text-[11px]">
              {objective.delivery_outcome_id}
            </MonoText>
          </span>
        ) : null}
      </div>
    </section>
  );
}
