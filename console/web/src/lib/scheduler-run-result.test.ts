import { describe, expect, test } from "vitest";

import {
  formatSchedulerTerminationReason,
  getSchedulerRunResultTone,
  getSchedulerRunResultView,
} from "./scheduler-run-result";

describe("scheduler-run-result helpers", () => {
  test("formats known termination reasons and derives tone", () => {
    expect(formatSchedulerTerminationReason("cancelled")).toBe("Cancelled");
    expect(
      getSchedulerRunResultTone({
        run_id: "run-1",
        termination_reason: "timeout",
        summary: "Timed out",
        error: null,
        completed_at: "2026-04-14T00:00:00Z",
      }),
    ).toBe("warning");
  });

  test("falls back to legacy summary when no structured result exists", () => {
    expect(getSchedulerRunResultView(null, "legacy summary")).toEqual({
      reasonLabel: null,
      tone: "neutral",
      summary: "legacy summary",
      error: null,
      message: "legacy summary",
      completedAt: null,
      runId: null,
    });
  });

  test("prefers explicit error over summary for message rendering", () => {
    expect(
      getSchedulerRunResultView(
        {
          run_id: "run-2",
          termination_reason: "error",
          summary: "step failed",
          error: "tool crashed",
          completed_at: "2026-04-14T00:00:00Z",
        },
        "legacy summary",
      ),
    ).toMatchObject({
      reasonLabel: "Errored",
      tone: "danger",
      summary: "step failed",
      error: "tool crashed",
      message: "tool crashed",
    });
  });
});
