import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  listAgents: vi.fn(),
  deleteAgent: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    listAgents: apiMocks.listAgents,
    deleteAgent: apiMocks.deleteAgent,
  };
});

import AgentsPage from "./page";

describe("AgentsPage", () => {
  test("marks the default agent, removes chat, and only shows delete on non-default agents", async () => {
    apiMocks.listAgents.mockResolvedValue([
      {
        id: "default-console-agent",
        name: "Walaha",
        description: "Default env-backed agent",
        model_provider: "openai-compatible",
        model_name: "qwen3.6-plus-preview",
        system_prompt: "",
        tools: ["bash", "web_search"],
        options: {
          config_root: "",
          max_steps: 50,
          run_timeout: 0,
          max_input_tokens_per_call: null,
          max_run_cost: null,
          enable_termination_summary: true,
          termination_summary_prompt: "",
          enable_skill: true,
          skills_dirs: null,
          relevant_memory_max_token: 0,
          stream_cleanup_timeout: 0,
          compact_prompt: "",
        },
        model_params: {
          base_url: "https://example.com",
          api_key_env_name: "API_KEY",
          max_output_tokens: 4096,
          max_context_window: 0,
          temperature: 1,
          top_p: 1,
          frequency_penalty: 0,
          presence_penalty: 0,
          cache_hit_price: 0,
          input_price: 0,
          output_price: 0,
        },
        created_at: "2026-04-02T00:00:00Z",
        updated_at: "2026-04-02T00:00:00Z",
        is_default: true,
      },
      {
        id: "agent-2",
        name: "Custom Agent",
        description: "",
        model_provider: "openai",
        model_name: "gpt-4o-mini",
        system_prompt: "",
        tools: [],
        options: {
          config_root: "",
          max_steps: 50,
          run_timeout: 0,
          max_input_tokens_per_call: null,
          max_run_cost: null,
          enable_termination_summary: true,
          termination_summary_prompt: "",
          enable_skill: true,
          skills_dirs: null,
          relevant_memory_max_token: 0,
          stream_cleanup_timeout: 0,
          compact_prompt: "",
        },
        model_params: {
          base_url: null,
          api_key_env_name: null,
          max_output_tokens: 4096,
          max_context_window: 0,
          temperature: 1,
          top_p: 1,
          frequency_penalty: 0,
          presence_penalty: 0,
          cache_hit_price: 0,
          input_price: 0,
          output_price: 0,
        },
        created_at: "2026-04-02T00:00:00Z",
        updated_at: "2026-04-02T00:00:00Z",
        is_default: false,
      },
    ]);

    render(<AgentsPage />);

    await waitFor(() => {
      expect(screen.getByText("Walaha")).toBeInTheDocument();
    });

    expect(screen.queryByText("Chat")).not.toBeInTheDocument();
    expect(screen.getByText("Default")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Delete Walaha" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Delete Custom Agent" }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByRole("link", { name: "Scheduler" }).length,
    ).toBeGreaterThan(0);
  });
});
