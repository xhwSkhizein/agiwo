"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import {
  AgentConfig,
  AgentConfigCreate,
  AvailableTool,
  listAvailableTools,
} from "@/lib/api";

const PROVIDERS = [
  { value: "openai", label: "OpenAI" },
  { value: "deepseek", label: "DeepSeek" },
  { value: "anthropic", label: "Anthropic" },
  { value: "openai-compatible", label: "OpenAI-compatible" },
  { value: "anthropic-compatible", label: "Anthropic-compatible" },
  { value: "nvidia", label: "Nvidia" },
  { value: "bedrock-anthropic", label: "Bedrock-anthropic" },
];

const COMPATIBLE_PROVIDERS = new Set([
  "openai-compatible",
  "anthropic-compatible",
]);

type AgentFormProps = {
  initialAgent?: AgentConfig | null;
  excludeAgentId?: string;
  submitLabel: string;
  submitting: boolean;
  error: string | null;
  onSubmit: (payload: AgentConfigCreate) => Promise<void>;
};

type AgentFormState = {
  name: string;
  description: string;
  modelProvider: string;
  modelName: string;
  baseUrl: string;
  apiKeyEnvName: string;
  systemPrompt: string;
  configRoot: string;
  maxSteps: number;
  runTimeout: number;
  maxInputTokensPerCall: string;
  maxRunCost: string;
  enableTerminationSummary: boolean;
  terminationSummaryPrompt: string;
  enableSkill: boolean;
  skillsDirsText: string;
  relevantMemoryMaxToken: number;
  streamCleanupTimeout: number;
  compactPrompt: string;
  maxOutputTokens: number;
  maxContextWindow: number;
  temperature: number;
  topP: number;
  frequencyPenalty: number;
  presencePenalty: number;
  cacheHitPrice: number;
  inputPrice: number;
  outputPrice: number;
  selectedTools: string[];
};

const DEFAULT_FORM_STATE: AgentFormState = {
  name: "",
  description: "",
  modelProvider: "deepseek",
  modelName: "deepseek-chat",
  baseUrl: "",
  apiKeyEnvName: "",
  systemPrompt: "",
  configRoot: "",
  maxSteps: 10,
  runTimeout: 600,
  maxInputTokensPerCall: "",
  maxRunCost: "",
  enableTerminationSummary: true,
  terminationSummaryPrompt: "",
  enableSkill: false,
  skillsDirsText: "",
  relevantMemoryMaxToken: 2048,
  streamCleanupTimeout: 300,
  compactPrompt: "",
  maxOutputTokens: 4096,
  maxContextWindow: 200000,
  temperature: 0.7,
  topP: 1.0,
  frequencyPenalty: 0.0,
  presencePenalty: 0.0,
  cacheHitPrice: 0,
  inputPrice: 0,
  outputPrice: 0,
  selectedTools: [],
};

function skillsDirsToText(value: string[] | null | undefined): string {
  return (value ?? []).join("\n");
}

function parseSkillsDirs(text: string): string[] | undefined {
  const entries = text
    .split(/\r?\n/)
    .map((entry) => entry.trim())
    .filter(Boolean);
  return entries.length > 0 ? entries : undefined;
}

function buildFormState(agent?: AgentConfig | null): AgentFormState {
  if (!agent) {
    return { ...DEFAULT_FORM_STATE };
  }

  return {
    name: agent.name,
    description: agent.description,
    modelProvider: agent.model_provider,
    modelName: agent.model_name,
    baseUrl: agent.model_params?.base_url ?? "",
    apiKeyEnvName: agent.model_params?.api_key_env_name ?? "",
    systemPrompt: agent.system_prompt,
    configRoot: agent.options?.config_root ?? "",
    maxSteps: agent.options?.max_steps ?? 10,
    runTimeout: agent.options?.run_timeout ?? 600,
    maxInputTokensPerCall:
      typeof agent.options?.max_input_tokens_per_call === "number"
        ? String(agent.options.max_input_tokens_per_call)
        : "",
    maxRunCost:
      typeof agent.options?.max_run_cost === "number"
        ? String(agent.options.max_run_cost)
        : "",
    enableTerminationSummary:
      agent.options?.enable_termination_summary ?? true,
    terminationSummaryPrompt: agent.options?.termination_summary_prompt ?? "",
    enableSkill: agent.options?.enable_skill ?? false,
    skillsDirsText: skillsDirsToText(agent.options?.skills_dirs),
    relevantMemoryMaxToken: agent.options?.relevant_memory_max_token ?? 2048,
    streamCleanupTimeout: agent.options?.stream_cleanup_timeout ?? 300,
    compactPrompt: agent.options?.compact_prompt ?? "",
    maxOutputTokens: agent.model_params?.max_output_tokens ?? 4096,
    maxContextWindow: agent.model_params?.max_context_window ?? 200000,
    temperature: agent.model_params?.temperature ?? 0.7,
    topP: agent.model_params?.top_p ?? 1.0,
    frequencyPenalty: agent.model_params?.frequency_penalty ?? 0.0,
    presencePenalty: agent.model_params?.presence_penalty ?? 0.0,
    cacheHitPrice: agent.model_params?.cache_hit_price ?? 0,
    inputPrice: agent.model_params?.input_price ?? 0,
    outputPrice: agent.model_params?.output_price ?? 0,
    selectedTools: agent.tools ?? [],
  };
}

export function AgentForm({
  initialAgent,
  excludeAgentId,
  submitLabel,
  submitting,
  error,
  onSubmit,
}: AgentFormProps) {
  const [form, setForm] = useState<AgentFormState>(() => buildFormState(initialAgent));
  const [availableTools, setAvailableTools] = useState<AvailableTool[]>([]);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    setForm(buildFormState(initialAgent));
  }, [initialAgent]);

  useEffect(() => {
    listAvailableTools(excludeAgentId).then(setAvailableTools).catch(() => { });
  }, [excludeAgentId]);

  const builtinTools = useMemo(
    () => availableTools.filter((tool) => tool.type === "builtin"),
    [availableTools]
  );
  const agentTools = useMemo(
    () => availableTools.filter((tool) => tool.type === "agent"),
    [availableTools]
  );

  const displayedError = localError ?? error;

  const setField = <K extends keyof AgentFormState>(
    key: K,
    value: AgentFormState[K]
  ) => {
    setLocalError(null);
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const toggleTool = (toolName: string) => {
    setLocalError(null);
    setForm((prev) => ({
      ...prev,
      selectedTools: prev.selectedTools.includes(toolName)
        ? prev.selectedTools.filter((tool) => tool !== toolName)
        : [...prev.selectedTools, toolName],
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name.trim()) {
      setLocalError("Name is required");
      return;
    }
    if (
      COMPATIBLE_PROVIDERS.has(form.modelProvider) &&
      form.baseUrl.trim() === ""
    ) {
      setLocalError("Compatible providers require a Base URL");
      return;
    }
    if (
      COMPATIBLE_PROVIDERS.has(form.modelProvider) &&
      form.apiKeyEnvName.trim() === ""
    ) {
      setLocalError("Compatible providers require an API Key Env Name");
      return;
    }

    await onSubmit({
      name: form.name.trim(),
      description: form.description,
      model_provider: form.modelProvider,
      model_name: form.modelName,
      system_prompt: form.systemPrompt,
      tools: form.selectedTools,
      options: {
        config_root: form.configRoot,
        max_steps: form.maxSteps,
        run_timeout: form.runTimeout,
        max_input_tokens_per_call:
          form.maxInputTokensPerCall.trim() === ""
            ? undefined
            : Number(form.maxInputTokensPerCall),
        max_run_cost:
          form.maxRunCost.trim() === ""
            ? undefined
            : Number(form.maxRunCost),
        enable_termination_summary: form.enableTerminationSummary,
        termination_summary_prompt: form.terminationSummaryPrompt,
        enable_skill: form.enableSkill,
        skills_dirs: parseSkillsDirs(form.skillsDirsText),
        relevant_memory_max_token: form.relevantMemoryMaxToken,
        stream_cleanup_timeout: form.streamCleanupTimeout,
        compact_prompt: form.compactPrompt,
      },
      model_params: {
        base_url: form.baseUrl.trim() === "" ? undefined : form.baseUrl.trim(),
        api_key_env_name:
          form.apiKeyEnvName.trim() === ""
            ? undefined
            : form.apiKeyEnvName.trim(),
        max_output_tokens: form.maxOutputTokens,
        max_context_window: form.maxContextWindow,
        temperature: form.temperature,
        top_p: form.topP,
        frequency_penalty: form.frequencyPenalty,
        presence_penalty: form.presencePenalty,
        cache_hit_price: form.cacheHitPrice,
        input_price: form.inputPrice,
        output_price: form.outputPrice,
      },
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div>
        <label className="block text-sm text-zinc-400 mb-1.5">Name *</label>
        <input
          type="text"
          value={form.name}
          onChange={(e) => setField("name", e.target.value)}
          className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          placeholder="My Assistant"
        />
      </div>

      <div>
        <label className="block text-sm text-zinc-400 mb-1.5">
          Description
        </label>
        <input
          type="text"
          value={form.description}
          onChange={(e) => setField("description", e.target.value)}
          className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          placeholder="A helpful assistant"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Model Provider
          </label>
          <select
            value={form.modelProvider}
            onChange={(e) => setField("modelProvider", e.target.value)}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          >
            {PROVIDERS.map((provider) => (
              <option key={provider.value} value={provider.value}>
                {provider.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Model Name
          </label>
          <input
            type="text"
            value={form.modelName}
            onChange={(e) => setField("modelName", e.target.value)}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            placeholder="deepseek-chat"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Base URL
          </label>
          <input
            type="text"
            value={form.baseUrl}
            onChange={(e) => setField("baseUrl", e.target.value)}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            placeholder="https://api.example.com/v1"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            API Key Env Name
          </label>
          <input
            type="text"
            value={form.apiKeyEnvName}
            onChange={(e) => setField("apiKeyEnvName", e.target.value)}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            placeholder="MINIMAX_API_KEY"
          />
        </div>
      </div>
      <p className="text-xs text-zinc-500">
        `openai-compatible` / `anthropic-compatible` should set both Base URL
        and API Key Env Name. Official providers can leave them empty to use
        provider defaults.
      </p>

      <div>
        <label className="block text-sm text-zinc-400 mb-1.5">
          System Prompt
        </label>
        <textarea
          value={form.systemPrompt}
          onChange={(e) => setField("systemPrompt", e.target.value)}
          rows={5}
          className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600 resize-y"
          placeholder="You are a helpful assistant..."
        />
      </div>

      <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">
        Builtin Tools
      </p>
      <div className="flex flex-wrap gap-2">
        {builtinTools.map((tool) => (
          <button
            key={tool.name}
            type="button"
            onClick={() => toggleTool(tool.name)}
            className={`px-3 py-1.5 rounded-md border text-sm transition-colors ${form.selectedTools.includes(tool.name)
                ? "bg-white text-black border-white"
                : "bg-zinc-900 text-zinc-400 border-zinc-700 hover:border-zinc-500"
              }`}
            title={tool.description}
          >
            {tool.name}
          </button>
        ))}
        {builtinTools.length === 0 && (
          <span className="text-xs text-zinc-600">No builtin tools available</span>
        )}
      </div>

      <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">
        Agent Tools
      </p>
      <div className="flex flex-wrap gap-2">
        {agentTools.map((tool) => (
          <button
            key={tool.name}
            type="button"
            onClick={() => toggleTool(tool.name)}
            className={`px-3 py-1.5 rounded-md border text-sm transition-colors ${form.selectedTools.includes(tool.name)
                ? "bg-blue-600 text-white border-blue-500"
                : "bg-zinc-900 text-zinc-400 border-zinc-700 hover:border-zinc-500"
              }`}
            title={tool.description}
          >
            {tool.agent_name || tool.name}
          </button>
        ))}
        {agentTools.length === 0 && (
          <span className="text-xs text-zinc-600">
            No other agents available as tools
          </span>
        )}
      </div>

      <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">
        Runtime
      </p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <label className="block text-sm text-zinc-400 mb-1.5">
            Config Root
          </label>
          <input
            type="text"
            value={form.configRoot}
            onChange={(e) => setField("configRoot", e.target.value)}
            placeholder="Optional override for workspace root"
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">Max Steps</label>
          <input
            type="number"
            value={form.maxSteps}
            onChange={(e) => setField("maxSteps", Number(e.target.value))}
            min={1}
            max={100}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Run Timeout (s)
          </label>
          <input
            type="number"
            value={form.runTimeout}
            onChange={(e) => setField("runTimeout", Number(e.target.value))}
            min={10}
            max={3600}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Max Input Tokens Per Call
          </label>
          <input
            type="number"
            value={form.maxInputTokensPerCall}
            onChange={(e) => setField("maxInputTokensPerCall", e.target.value)}
            min={1}
            max={512000}
            placeholder="Auto (max_context_window - max_output_tokens)"
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Max Run Cost (USD)
          </label>
          <input
            type="number"
            value={form.maxRunCost}
            onChange={(e) => setField("maxRunCost", e.target.value)}
            min={0}
            step={0.000001}
            placeholder="Optional"
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Relevant Memory Max Token
          </label>
          <input
            type="number"
            value={form.relevantMemoryMaxToken}
            onChange={(e) =>
              setField("relevantMemoryMaxToken", Number(e.target.value))
            }
            min={1}
            max={32768}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Stream Cleanup Timeout (s)
          </label>
          <input
            type="number"
            value={form.streamCleanupTimeout}
            onChange={(e) =>
              setField("streamCleanupTimeout", Number(e.target.value))
            }
            min={1}
            step={0.1}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
      </div>

      <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">
        Skills
      </p>
      <div className="space-y-3">
        <label className="flex items-center gap-2 text-sm text-zinc-400 cursor-pointer">
          <input
            type="checkbox"
            checked={form.enableSkill}
            onChange={(e) => setField("enableSkill", e.target.checked)}
            className="rounded border-zinc-700 bg-zinc-900"
          />
          Enable Skills
        </label>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Skill Directories
          </label>
          <textarea
            value={form.skillsDirsText}
            onChange={(e) => setField("skillsDirsText", e.target.value)}
            rows={3}
            placeholder={"One directory per line\nskills\n~/.agent/skills"}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600 resize-y"
          />
          <p className="mt-1 text-xs text-zinc-500">
            Leave empty to use global and default skill directories.
          </p>
        </div>
      </div>

      <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">
        Summaries & Compact
      </p>
      <div className="space-y-3">
        <label className="flex items-center gap-2 text-sm text-zinc-400 cursor-pointer">
          <input
            type="checkbox"
            checked={form.enableTerminationSummary}
            onChange={(e) =>
              setField("enableTerminationSummary", e.target.checked)
            }
            className="rounded border-zinc-700 bg-zinc-900"
          />
          Enable Termination Summary
        </label>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Termination Summary Prompt
          </label>
          <textarea
            value={form.terminationSummaryPrompt}
            onChange={(e) =>
              setField("terminationSummaryPrompt", e.target.value)
            }
            rows={3}
            placeholder="Optional custom prompt for termination summary"
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600 resize-y"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Compact Prompt
          </label>
          <textarea
            value={form.compactPrompt}
            onChange={(e) => setField("compactPrompt", e.target.value)}
            rows={4}
            placeholder="Optional custom prompt used during context compact"
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600 resize-y"
          />
        </div>
      </div>

      <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">
        Model Parameters
      </p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Max Output Tokens
          </label>
          <input
            type="number"
            value={form.maxOutputTokens}
            onChange={(e) =>
              setField("maxOutputTokens", Number(e.target.value))
            }
            min={1}
            max={128000}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Max Context Window
          </label>
          <input
            type="number"
            value={form.maxContextWindow}
            onChange={(e) =>
              setField("maxContextWindow", Number(e.target.value))
            }
            min={1}
            max={512000}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Temperature
          </label>
          <input
            type="number"
            value={form.temperature}
            onChange={(e) => setField("temperature", Number(e.target.value))}
            min={0}
            max={2}
            step={0.1}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">Top P</label>
          <input
            type="number"
            value={form.topP}
            onChange={(e) => setField("topP", Number(e.target.value))}
            min={0}
            max={1}
            step={0.01}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Frequency Penalty
          </label>
          <input
            type="number"
            value={form.frequencyPenalty}
            onChange={(e) =>
              setField("frequencyPenalty", Number(e.target.value))
            }
            min={-2}
            max={2}
            step={0.1}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Presence Penalty
          </label>
          <input
            type="number"
            value={form.presencePenalty}
            onChange={(e) =>
              setField("presencePenalty", Number(e.target.value))
            }
            min={-2}
            max={2}
            step={0.1}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Cache-Hit Price (USD / 1M Tokens)
          </label>
          <input
            type="number"
            value={form.cacheHitPrice}
            onChange={(e) => setField("cacheHitPrice", Number(e.target.value))}
            min={0}
            step={0.000001}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Input Price (USD / 1M Tokens)
          </label>
          <input
            type="number"
            value={form.inputPrice}
            onChange={(e) => setField("inputPrice", Number(e.target.value))}
            min={0}
            step={0.000001}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">
            Output Price (USD / 1M Tokens)
          </label>
          <input
            type="number"
            value={form.outputPrice}
            onChange={(e) => setField("outputPrice", Number(e.target.value))}
            min={0}
            step={0.000001}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>
      </div>

      {displayedError && <p className="text-sm text-red-400">{displayedError}</p>}

      <div className="flex gap-3 pt-2">
        <button
          type="submit"
          disabled={submitting}
          className="px-5 py-2 rounded-md bg-white text-black text-sm font-medium hover:bg-zinc-200 transition-colors disabled:opacity-50"
        >
          {submitting ? "Saving..." : submitLabel}
        </button>
        <Link
          href="/agents"
          className="px-5 py-2 rounded-md border border-zinc-700 text-sm hover:bg-zinc-800 transition-colors"
        >
          Cancel
        </Link>
      </div>
    </form>
  );
}
