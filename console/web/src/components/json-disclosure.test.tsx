import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

import { JsonDisclosure } from "./json-disclosure";

describe("JsonDisclosure", () => {
  test("copies the serialized JSON without expanding first", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    render(<JsonDisclosure label="Raw step JSON" value={{ role: "assistant" }} />);

    fireEvent.click(screen.getByRole("button", { name: "Copy Raw step JSON" }));

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith('{\n  "role": "assistant"\n}');
    });
    expect(screen.getByText("Copied")).toBeInTheDocument();
  });
});
