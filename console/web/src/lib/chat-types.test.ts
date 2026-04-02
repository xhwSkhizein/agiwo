import { describe, expect, test } from "vitest";

import {
  appendUnseenStepMessages,
  contentToText,
  messageFromStep,
  messagesFromSteps,
} from "./chat-types";
import type { StepResponse } from "./api";

function buildStep(partial: Partial<StepResponse>): StepResponse {
  return {
    id: "step-1",
    session_id: "sess-1",
    run_id: "run-1",
    sequence: 1,
    role: "assistant",
    agent_id: "agent-1",
    content: null,
    content_for_user: null,
    reasoning_content: null,
    user_input: null,
    tool_calls: null,
    tool_call_id: null,
    name: null,
    metrics: null,
    created_at: "2026-04-02T00:00:00Z",
    parent_run_id: null,
    depth: 0,
    ...partial,
  };
}

describe("chat-types", () => {
  test("preserves structured user input without flattening it into plain text", () => {
    const step = buildStep({
      role: "user",
      user_input: {
        __type: "user_message",
        content: [
          { type: "text", text: "Plan the task" },
          { type: "image", url: "https://example.com/mock.png" },
        ],
        context: {
          source: "console",
          metadata: { channel: "scheduler-tree" },
        },
      },
    });

    const message = messageFromStep(step);

    expect(message).toMatchObject({
      role: "user",
      userInput: step.user_input,
      structuredContent: step.user_input,
      rawContent: step.user_input,
    });
    expect(message?.text).toBeUndefined();
  });

  test("prefers structured content text before content_for_user and retains reasoning/tool calls", () => {
    const step = buildStep({
      content: [
        { type: "text", text: "Primary assistant output" },
        { type: "text", text: "Second line" },
      ],
      content_for_user: "fallback output",
      reasoning_content: "thinking",
      tool_calls: [
        {
          id: "tool-1",
          type: "function",
          function: {
            name: "search",
            arguments: "{\"q\":\"agiwo\"}",
          },
        },
      ],
    });

    const message = messageFromStep(step);

    expect(message).toMatchObject({
      role: "assistant",
      text: "Primary assistant output\nSecond line",
      reasoningContent: "thinking",
      toolCalls: step.tool_calls,
      rawContent: step.content,
    });
  });

  test("filters empty assistant records when a step carries no visible content", () => {
    const messages = messagesFromSteps([
      buildStep({ id: "empty-1" }),
      buildStep({ id: "assistant-2", content_for_user: "Visible output" }),
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]?.text).toBe("Visible output");
  });

  test("extracts text content from content parts arrays", () => {
    expect(
      contentToText([
        { type: "text", text: "hello" },
        { type: "image", url: "https://example.com/image.png" },
        { type: "text", text: "world" },
      ]),
    ).toBe("hello\nworld");
  });

  test("appends only unseen step-backed messages during reconciliation", () => {
    const existing = messagesFromSteps([
      buildStep({ id: "step-1", sequence: 1, content_for_user: "first" }),
    ]);

    const merged = appendUnseenStepMessages(existing, [
      buildStep({ id: "step-1", sequence: 1, content_for_user: "first" }),
      buildStep({ id: "step-2", sequence: 2, content_for_user: "second" }),
    ]);

    expect(merged).toHaveLength(2);
    expect(merged.map((message) => message.text)).toEqual(["first", "second"]);
  });
});
