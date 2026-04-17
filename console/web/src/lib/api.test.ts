import { afterEach, describe, expect, test, vi } from "vitest";

afterEach(() => {
  delete process.env.NEXT_PUBLIC_API_URL;
  vi.resetModules();
});

describe("api url base", () => {
  test("defaults to same-origin relative paths", async () => {
    delete process.env.NEXT_PUBLIC_API_URL;

    const api = await import("./api");

    expect(api.sessionInputStreamUrl("sess-1")).toBe("/api/sessions/sess-1/input");
  });

  test("uses NEXT_PUBLIC_API_URL when configured", async () => {
    process.env.NEXT_PUBLIC_API_URL = "http://localhost:8422/";

    const api = await import("./api");

    expect(api.sessionInputStreamUrl("sess-1")).toBe(
      "http://localhost:8422/api/sessions/sess-1/input",
    );
  });
});
