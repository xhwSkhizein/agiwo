import type { StepResponse, ToolCallPayload, UserInput } from "./api";

export type ChatRole = "user" | "assistant" | "tool" | "system";

export interface ChatMessage {
  id: string;
  stepId?: string;
  role: ChatRole;
  sequence?: number;
  sourceAgentId?: string;
  userInput?: UserInput | null;
  text?: string;
  originalContent?: string;
  structuredContent?: unknown;
  rawContent?: unknown;
  toolCalls?: ToolCallPayload[];
  reasoningContent?: string;
  name?: string;
  isStreaming?: boolean;
}

export function genMessageId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function contentToText(value: unknown): string | undefined {
  if (typeof value === "string") {
    return value;
  }
  if (value === null || value === undefined) {
    return undefined;
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((item) =>
        item && typeof item === "object" && "text" in item && typeof item.text === "string"
          ? item.text
          : "",
      )
      .filter(Boolean);
    return parts.length > 0 ? parts.join("\n") : undefined;
  }
  return undefined;
}

export function messageFromStep(step: StepResponse): ChatMessage | null {
  if (step.role === "user") {
    if (!step.user_input) {
      return null;
    }
    return {
      id: genMessageId(),
      stepId: step.id,
      role: "user",
      sequence: step.sequence,
      sourceAgentId: step.agent_id ?? undefined,
      userInput: step.user_input,
      text: typeof step.user_input === "string" ? step.user_input : undefined,
      structuredContent:
        typeof step.user_input === "string" ? undefined : step.user_input,
      rawContent: step.user_input,
    };
  }

  if (step.role === "assistant") {
    const text = contentToText(step.content) ?? step.content_for_user ?? undefined;
    if (!text && !step.tool_calls?.length && !step.reasoning_content && !step.content) {
      return null;
    }
    return {
      id: genMessageId(),
      stepId: step.id,
      role: "assistant",
      sequence: step.sequence,
      sourceAgentId: step.agent_id ?? undefined,
      text,
      structuredContent:
        typeof step.content === "string" ? undefined : step.content ?? undefined,
      rawContent: step.content ?? undefined,
      toolCalls: step.tool_calls ?? undefined,
      reasoningContent: step.reasoning_content ?? undefined,
    };
  }

  if (step.role === "tool") {
    const hasCondensed = typeof step.condensed_content === "string";
    return {
      id: genMessageId(),
      stepId: step.id,
      role: "tool",
      sequence: step.sequence,
      sourceAgentId: step.agent_id ?? undefined,
      text: hasCondensed
        ? step.condensed_content!
        : (contentToText(step.content) ?? undefined),
      originalContent: hasCondensed
        ? (contentToText(step.content) ?? undefined)
        : undefined,
      structuredContent:
        typeof step.content === "string" ? undefined : step.content ?? undefined,
      rawContent: step.content ?? undefined,
      name: step.name ?? undefined,
    };
  }

  return null;
}

export function messagesFromSteps(steps: StepResponse[]): ChatMessage[] {
  return steps
    .map(messageFromStep)
    .filter((message): message is ChatMessage => message !== null);
}

export function appendUnseenStepMessages(
  existing: ChatMessage[],
  steps: StepResponse[],
): ChatMessage[] {
  const seenStepIds = new Set(
    existing
      .map((message) => message.stepId)
      .filter((stepId): stepId is string => Boolean(stepId)),
  );
  const appended: ChatMessage[] = [];
  for (const message of messagesFromSteps(steps)) {
    if (message.stepId && seenStepIds.has(message.stepId)) {
      continue;
    }
    if (message.stepId) {
      seenStepIds.add(message.stepId);
    }
    appended.push(message);
  }
  return [...existing, ...appended];
}

export function getHighestMessageSequence(messages: ChatMessage[]): number | null {
  let maxSequence: number | null = null;
  for (const message of messages) {
    if (typeof message.sequence !== "number") {
      continue;
    }
    if (maxSequence === null || message.sequence > maxSequence) {
      maxSequence = message.sequence;
    }
  }
  return maxSequence;
}
