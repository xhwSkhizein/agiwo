import type { ToolCallPayload } from "./api";

export type ChatRole = "user" | "assistant" | "tool" | "system";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  name?: string;
  tool_calls?: ToolCallPayload[];
  reasoning_content?: string;
  isStreaming?: boolean;
  agentId?: string;
}

export function genMessageId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
