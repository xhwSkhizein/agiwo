import { afterEach, describe, expect, test, vi } from "vitest";

afterEach(() => {
  delete process.env.NEXT_PUBLIC_API_URL;
  vi.useRealTimers();
  vi.unstubAllGlobals();
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

describe("fetch timeout and abort handling", () => {
  test("passes the internal timeout signal and cascades caller aborts", async () => {
    const callerController = new AbortController();
    let fetchSignal: AbortSignal | undefined;
    const abortError = new DOMException("Aborted", "AbortError");
    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string | URL | Request, init?: RequestInit) => {
        fetchSignal = init?.signal;
        return new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(abortError));
        });
      }),
    );
    const api = await import("./api");

    const request = api.getAgentState("state-1", {
      signal: callerController.signal,
    });
    callerController.abort();

    await expect(request).rejects.toBe(abortError);
    expect(fetchSignal).toBeDefined();
    expect(fetchSignal).not.toBe(callerController.signal);
    expect(fetchSignal?.aborted).toBe(true);
  });

  test("reports timeout only for the internal timeout abort", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string | URL | Request, init?: RequestInit) => {
        return new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("Aborted", "AbortError"));
          });
        });
      }),
    );
    const api = await import("./api");

    const request = api.getAgentState("state-1");
    const expectation = expect(request).rejects.toThrow("Request timed out after 30s");
    await vi.advanceTimersByTimeAsync(30_000);

    await expectation;
  });
});
