import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

import { useChatStream } from "./use-chat-stream";

function streamFromEvents(events: object[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(event)}\n\n`),
        );
      }
      controller.close();
    },
  });
}

describe("useChatStream", () => {
  test("streams assistant output and captures session/root identifiers", async () => {
    const onSessionCaptured = vi.fn();
    const onRootStateCaptured = vi.fn();

    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(
        streamFromEvents([
          {
            type: "scheduler_ack",
            session_id: "sess-1",
            state_id: "root-1",
          },
          {
            type: "run_started",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
          },
          {
            type: "step_delta",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            delta: { content: "Hello " },
          },
          {
            type: "step_delta",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            delta: { content: "world" },
          },
          {
            type: "run_completed",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            response: "done",
          },
        ]),
      ),
    );

    const { result } = renderHook(() =>
      useChatStream("/api/chat/agent-1", {
        onSessionCaptured,
        onRootStateCaptured,
      }),
    );

    await act(async () => {
      await result.current.sendMessage("Hi");
    });

    await waitFor(() => {
      expect(result.current.isStreaming).toBe(false);
    });

    expect(onSessionCaptured).toHaveBeenCalledWith("sess-1");
    expect(onRootStateCaptured).toHaveBeenCalledWith("root-1");
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1]).toMatchObject({
      role: "assistant",
      text: "Hello world",
      isStreaming: false,
    });
  });

  test("classifies 404 stream failures instead of surfacing a generic HTTP code only", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(null, {
        status: 404,
        statusText: "Not Found",
      }),
    );

    const { result } = renderHook(() => useChatStream("/api/chat/agent-1"));

    await act(async () => {
      await result.current.sendMessage("Hi");
    });

    await waitFor(() => {
      expect(result.current.isStreaming).toBe(false);
    });

    expect(result.current.messages.at(-1)?.text).toBe(
      "Error: Requested session or agent was not found.",
    );
  });

  test("creates a fresh assistant message when a continuous stream starts a second root run", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(
        streamFromEvents([
          {
            type: "run_started",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
          },
          {
            type: "step_delta",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            delta: { content: "Waiting on child" },
          },
          {
            type: "run_completed",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            response: "sleep",
            termination_reason: "sleeping",
          },
          {
            type: "run_started",
            session_id: "sess-1",
            run_id: "run-2",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
          },
          {
            type: "step_delta",
            session_id: "sess-1",
            run_id: "run-2",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            delta: { content: "Final answer" },
          },
          {
            type: "run_completed",
            session_id: "sess-1",
            run_id: "run-2",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            response: "done",
          },
        ]),
      ),
    );

    const { result } = renderHook(() => useChatStream("/api/sessions/sess-1/input"));

    await act(async () => {
      await result.current.sendMessage("Hi");
    });

    await waitFor(() => {
      expect(result.current.isStreaming).toBe(false);
    });

    const assistantMessages = result.current.messages.filter(
      (message) => message.role === "assistant",
    );
    expect(assistantMessages).toHaveLength(2);
    expect(assistantMessages[0]?.text).toBe("Waiting on child");
    expect(assistantMessages[1]?.text).toBe("Final answer");
  });

  test("does not leave a dangling streaming assistant bubble after consecutive tool steps", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(
        streamFromEvents([
          {
            type: "run_started",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
          },
          {
            type: "step_completed",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            step: {
              id: "assistant-step-1",
              sequence: 2,
              role: "assistant",
              agent_id: "root-1",
              content: null,
              content_for_user: null,
              tool_calls: [
                {
                  id: "tool-1",
                  type: "function",
                  function: { name: "spawn_agent", arguments: "{\"task\":\"calc\"}" },
                },
              ],
              reasoning_content: null,
            },
          },
          {
            type: "step_completed",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            step: {
              id: "tool-step-1",
              sequence: 3,
              role: "tool",
              agent_id: "root-1",
              name: "spawn_agent",
              content: "spawned",
            },
          },
          {
            type: "step_completed",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            step: {
              id: "tool-step-2",
              sequence: 4,
              role: "tool",
              agent_id: "root-1",
              name: "sleep_and_wait",
              content: "waiting",
            },
          },
          {
            type: "run_completed",
            session_id: "sess-1",
            run_id: "run-1",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            response: "sleep",
            termination_reason: "sleeping",
          },
          {
            type: "run_started",
            session_id: "sess-1",
            run_id: "run-2",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
          },
          {
            type: "step_completed",
            session_id: "sess-1",
            run_id: "run-2",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            step: {
              id: "assistant-step-2",
              sequence: 5,
              role: "assistant",
              agent_id: "root-1",
              content: "final answer",
              content_for_user: "final answer",
              tool_calls: null,
              reasoning_content: null,
            },
          },
          {
            type: "run_completed",
            session_id: "sess-1",
            run_id: "run-2",
            agent_id: "root-1",
            parent_run_id: null,
            depth: 0,
            response: "done",
          },
        ]),
      ),
    );

    const { result } = renderHook(() => useChatStream("/api/sessions/sess-1/input"));

    await act(async () => {
      await result.current.sendMessage("Hi");
    });

    await waitFor(() => {
      expect(result.current.isStreaming).toBe(false);
    });

    const assistantMessages = result.current.messages.filter(
      (message) => message.role === "assistant",
    );
    expect(assistantMessages.some((message) => message.isStreaming)).toBe(false);
  });
});
