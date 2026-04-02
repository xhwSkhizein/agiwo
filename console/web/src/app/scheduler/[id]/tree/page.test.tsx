import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

const mockReplace = vi.fn();

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "root-1" }),
  useRouter: () => ({ replace: mockReplace }),
  useSearchParams: () =>
    new URLSearchParams("selected=child-1"),
}));

vi.mock("@/components/scheduler-tree/scheduler-tree-workspace", () => ({
  SchedulerTreeWorkspace: ({
    rootStateId,
    selectedStateId,
  }: {
    rootStateId: string;
    selectedStateId: string | null;
  }) => (
    <div>
      <span data-testid="root-state">{rootStateId}</span>
      <span data-testid="selected-state">{selectedStateId}</span>
    </div>
  ),
}));

import SchedulerTreePage from "./page";

describe("SchedulerTreePage", () => {
  test("passes the route id and selected query param into the workspace", () => {
    render(<SchedulerTreePage />);

    expect(screen.getByTestId("root-state")).toHaveTextContent("root-1");
    expect(screen.getByTestId("selected-state")).toHaveTextContent("child-1");
  });
});
