"use client";

import Link from "next/link";
import { useEffect, useId, useMemo, useState, type ReactNode } from "react";

import { ErrorStateMessage } from "@/components/state-message";
import { PillBadge } from "@/components/pill-badge";
import { cn } from "@/lib/utils";
import {
  AgentCapabilities,
  AgentConfig,
  AgentConfigCreate,
  AgentProviderCapability,
  AvailableSkill,
  AvailableTool,
  getAgentCapabilities,
  listAvailableSkills,
  listAvailableTools,
} from "@/lib/api";

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
  relevantMemoryMaxToken: number;
  streamCleanupTimeout: number;
  compactPrompt: string;
  enableContextRollback: boolean;
  enableToolRetrospect: boolean;
  retrospectTokenThreshold: number;
  retrospectRoundInterval: number;
  retrospectAccumulatedTokenThreshold: number;
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
  selectedSkills: string[];
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
  relevantMemoryMaxToken: 2048,
  streamCleanupTimeout: 300,
  compactPrompt: "",
  enableContextRollback: true,
  enableToolRetrospect: true,
  retrospectTokenThreshold: 1024,
  retrospectRoundInterval: 5,
  retrospectAccumulatedTokenThreshold: 8192,
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
  selectedSkills: [],
};

/**
 * Create an AgentFormState populated from an existing AgentConfig or default values.
 *
 * When `agent` is provided, maps its fields into the form-shaped representation, converting
 * model/option parameters into the form-friendly types and applying sensible defaults for
 * missing values. When `agent` is not provided, returns a shallow copy of DEFAULT_FORM_STATE.
 *
 * @param agent - Optional source AgentConfig to populate the form state from
 * @returns An AgentFormState reflecting `agent`'s values with fallbacks; certain numeric option
 * fields are converted to strings where the form expects text input (e.g., `maxInputTokensPerCall`, `maxRunCost`).
 */
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
    relevantMemoryMaxToken: agent.options?.relevant_memory_max_token ?? 2048,
    streamCleanupTimeout: agent.options?.stream_cleanup_timeout ?? 300,
    compactPrompt: agent.options?.compact_prompt ?? "",
    enableContextRollback: agent.options?.enable_context_rollback ?? true,
    enableToolRetrospect: agent.options?.enable_tool_retrospect ?? true,
    retrospectTokenThreshold: agent.options?.retrospect_token_threshold ?? 1024,
    retrospectRoundInterval: agent.options?.retrospect_round_interval ?? 5,
    retrospectAccumulatedTokenThreshold:
      agent.options?.retrospect_accumulated_token_threshold ?? 8192,
    maxOutputTokens: agent.model_params?.max_output_tokens ?? 4096,
    maxContextWindow: agent.model_params?.max_context_window ?? 200000,
    temperature: agent.model_params?.temperature ?? 0.7,
    topP: agent.model_params?.top_p ?? 1.0,
    frequencyPenalty: agent.model_params?.frequency_penalty ?? 0.0,
    presencePenalty: agent.model_params?.presence_penalty ?? 0.0,
    cacheHitPrice: agent.model_params?.cache_hit_price ?? 0,
    inputPrice: agent.model_params?.input_price ?? 0,
    outputPrice: agent.model_params?.output_price ?? 0,
    selectedTools: agent.allowed_tools ?? [],
    selectedSkills: agent.allowed_skills ?? [],
  };
}

type FieldProps = {
  id: string;
  label: string;
  hint?: ReactNode;
  required?: boolean;
  children: ReactNode;
};

/**
 * Renders a labeled form field wrapper with an optional required marker and hint.
 *
 * @param id - The HTML id used for the associated input element and for hint aria linking.
 * @param label - The visible label text shown for the field.
 * @param hint - Optional explanatory text shown beneath the field.
 * @param required - If `true`, displays a required marker next to the label.
 * @param children - The input or control elements to render inside the field wrapper.
 * @returns A JSX element containing the label, the provided children, and the optional hint.
 */
function Field({ id, label, hint, required = false, children }: FieldProps) {
  const hintId = hint ? `${id}-hint` : undefined;

  return (
    <div>
      <label htmlFor={id} className="ui-field-label">
        {label}
        {required && <span className="ml-1 text-danger">*</span>}
      </label>
      {children}
      {hint && (
        <p id={hintId} className="ui-field-hint">
          {hint}
        </p>
      )}
    </div>
  );
}

type DisclosureSectionProps = {
  title: string;
  description: string;
  open?: boolean;
  children: ReactNode;
};

/**
 * Renders a collapsible section with a title, descriptive subtitle, and content.
 *
 * @param open - If `true`, the section is expanded. If `false` or `undefined`, the section is collapsed.
 * @returns A `<details>` element containing the section header (title and description) and the provided children.
 */
function DisclosureSection({
  title,
  description,
  open,
  children,
}: DisclosureSectionProps) {
  return (
    <details
      className="rounded-2xl border border-line bg-panel p-4"
      open={open || undefined}
    >
      <summary className="cursor-pointer list-none">
        <div className="space-y-1">
          <div className="ui-section-title">{title}</div>
          <p className="ui-section-copy text-sm">{description}</p>
        </div>
      </summary>
      <div className="mt-4 space-y-4">{children}</div>
    </details>
  );
}

type ToggleCardProps = {
  id: string;
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
};

/**
 * A labeled checkbox styled as a selectable card for toggling a boolean option.
 *
 * @param id - The HTML id applied to the underlying checkbox input.
 * @param label - Primary text displayed for the card.
 * @param description - Secondary explanatory text shown under the label.
 * @param checked - Whether the checkbox is currently checked.
 * @param onChange - Callback invoked with the new checked state when the user toggles the checkbox.
 * @returns The rendered toggle card element.
 */
function ToggleCard({
  id,
  label,
  description,
  checked,
  onChange,
}: ToggleCardProps) {
  return (
    <label className="flex items-start gap-3 rounded-2xl border border-line bg-panel p-3">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 h-4 w-4 rounded border-line bg-panel-strong text-accent"
      />
      <span className="space-y-1">
        <span className="block text-sm font-medium text-foreground">{label}</span>
        <span className="block text-xs text-ink-muted">{description}</span>
      </span>
    </label>
  );
}

type ToolToggleProps = {
  tool: AvailableTool;
  selected: boolean;
  onToggle: () => void;
};

/**
 * Renders a toggleable chip button representing a tool.
 *
 * Displays the tool's label and a meta string ("Agent tool" or "Builtin tool"), indicates its selected state, and calls `onToggle` when clicked.
 *
 * @param tool - The tool to render; its `agent_name` (preferred) or `name` is shown and `type` determines the meta label.
 * @param selected - Whether the tool is currently selected; reflected in the button's pressed/selected state.
 * @param onToggle - Click handler invoked to toggle the tool's selection.
 *
 * @returns A button element that represents and toggles the provided tool.
 */
function ToolToggle({ tool, selected, onToggle }: ToolToggleProps) {
  return (
    <button
      type="button"
      title={tool.description}
      aria-pressed={selected}
      data-selected={selected}
      onClick={onToggle}
      className="ui-chip-toggle"
    >
      <span className="font-medium">{tool.agent_name || tool.name}</span>
      <span className="ui-chip-toggle__meta">
        {tool.type === "agent" ? "Agent tool" : "Builtin tool"}
      </span>
    </button>
  );
}

/**
 * Render a form UI for creating or editing an agent configuration.
 *
 * The form loads available tools and provider capabilities, maintains local editable state,
 * validates required fields on submit, and converts inputs into an `AgentConfigCreate`
 * payload passed to `onSubmit`.
 *
 * @param initialAgent - Optional existing agent used to populate the form for editing.
 * @param excludeAgentId - Optional agent ID to exclude from available-agent tool choices.
 * @param submitLabel - Label for the primary submit button.
 * @param submitting - When `true`, the submit button is disabled and shows a saving state.
 * @param error - Optional external error message to display in the form.
 * @param onSubmit - Callback invoked with the constructed `AgentConfigCreate` payload when the form is submitted.
 * @returns A React element rendering the agent configuration form.
 */
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
  const [availableSkills, setAvailableSkills] = useState<AvailableSkill[]>([]);
  const [capabilities, setCapabilities] = useState<AgentCapabilities | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const formId = useId();

  useEffect(() => {
    setForm(buildFormState(initialAgent));
  }, [initialAgent]);

  useEffect(() => {
    listAvailableTools(excludeAgentId).then(setAvailableTools).catch(() => {});
  }, [excludeAgentId]);

  useEffect(() => {
    listAvailableSkills().then(setAvailableSkills).catch(() => {});
  }, []);

  useEffect(() => {
    getAgentCapabilities().then(setCapabilities).catch(() => {});
  }, []);

  const builtinTools = useMemo(
    () => availableTools.filter((tool) => tool.type === "builtin"),
    [availableTools],
  );
  const agentTools = useMemo(
    () => availableTools.filter((tool) => tool.type === "agent"),
    [availableTools],
  );

  const displayedError = localError ?? error;
  const providerCapability: AgentProviderCapability | undefined =
    capabilities?.providers.find((provider) => provider.value === form.modelProvider);

  const setField = <K extends keyof AgentFormState>(
    key: K,
    value: AgentFormState[K],
  ) => {
    setLocalError(null);
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const setBoundedIntegerField = (
    key:
      | "retrospectTokenThreshold"
      | "retrospectRoundInterval"
      | "retrospectAccumulatedTokenThreshold",
    rawValue: string,
    min: number,
    max: number,
  ) => {
    if (rawValue.trim() === "") {
      return;
    }
    const parsed = Number.parseInt(rawValue, 10);
    if (Number.isNaN(parsed)) {
      return;
    }
    setField(key, Math.min(max, Math.max(min, parsed)));
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

  const toggleSkill = (skillName: string) => {
    setLocalError(null);
    setForm((prev) => ({
      ...prev,
      selectedSkills: prev.selectedSkills.includes(skillName)
        ? prev.selectedSkills.filter((s) => s !== skillName)
        : [...prev.selectedSkills, skillName],
    }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!form.name.trim()) {
      setLocalError("Name is required");
      return;
    }
    if (providerCapability?.requires_base_url && form.baseUrl.trim() === "") {
      setLocalError("Compatible providers require a Base URL");
      return;
    }
    if (
      providerCapability?.requires_api_key_env_name &&
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
      allowed_tools: form.selectedTools,
      allowed_skills: form.selectedSkills,
      options: {
        config_root: form.configRoot,
        max_steps: form.maxSteps,
        run_timeout: form.runTimeout,
        max_input_tokens_per_call:
          form.maxInputTokensPerCall.trim() === ""
            ? null
            : Number(form.maxInputTokensPerCall),
        max_run_cost:
          form.maxRunCost.trim() === ""
            ? null
            : Number(form.maxRunCost),
        enable_termination_summary: form.enableTerminationSummary,
        termination_summary_prompt: form.terminationSummaryPrompt,
        relevant_memory_max_token: form.relevantMemoryMaxToken,
        stream_cleanup_timeout: form.streamCleanupTimeout,
        compact_prompt: form.compactPrompt,
        enable_context_rollback: form.enableContextRollback,
        enable_tool_retrospect: form.enableToolRetrospect,
        retrospect_token_threshold: form.retrospectTokenThreshold,
        retrospect_round_interval: form.retrospectRoundInterval,
        retrospect_accumulated_token_threshold:
          form.retrospectAccumulatedTokenThreshold,
      },
      model_params: {
        base_url: form.baseUrl.trim() === "" ? null : form.baseUrl.trim(),
        api_key_env_name:
          form.apiKeyEnvName.trim() === ""
            ? null
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

  const fieldId = (name: string) => `${formId}-${name}`;
  const hasCompatibilitySettings =
    form.baseUrl.trim() !== "" || form.apiKeyEnvName.trim() !== "";
  const compatibilityRequired =
    providerCapability?.requires_base_url ||
    providerCapability?.requires_api_key_env_name;
  const openCompatibilitySection = compatibilityRequired || hasCompatibilitySettings;
  const openWorkflowSection =
    form.terminationSummaryPrompt.trim() !== "" ||
    form.compactPrompt.trim() !== "" ||
    !form.enableContextRollback ||
    !form.enableToolRetrospect;

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      <div className="space-y-2">
        <p className="ui-section-kicker">Agent Setup</p>
        <h2 className="text-2xl font-semibold text-foreground">Start with the essentials</h2>
        <p className="ui-section-copy max-w-2xl">
          Name the agent, pick the model, choose the tools it can use, and leave the
          runtime tuning collapsed until you actually need it.
        </p>
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <PillBadge variant="default">{form.modelProvider}</PillBadge>
          <PillBadge variant="info">{form.selectedTools.length} tools selected</PillBadge>
          {form.selectedSkills.length > 0 && (
            <PillBadge variant="success">{form.selectedSkills.length} skills</PillBadge>
          )}
        </div>
      </div>

      {displayedError && <ErrorStateMessage>{displayedError}</ErrorStateMessage>}

      <section className="space-y-5">
        <div className="space-y-1">
          <p className="ui-section-kicker">Basics</p>
          <p className="ui-section-copy">
            These are the only fields required to create a working agent.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <Field
            id={fieldId("name")}
            label="Name"
            required={true}
            hint="Use a stable name that operators can recognize in traces and sessions."
          >
            <input
              id={fieldId("name")}
              type="text"
              value={form.name}
              onChange={(event) => setField("name", event.target.value)}
              className="ui-input"
              placeholder="Support triage agent"
              aria-invalid={displayedError === "Name is required"}
            />
          </Field>

          <Field
            id={fieldId("description")}
            label="Description"
            hint="Optional. Keep it short and operational."
          >
            <input
              id={fieldId("description")}
              type="text"
              value={form.description}
              onChange={(event) => setField("description", event.target.value)}
              className="ui-input"
              placeholder="Handles scheduling and session triage"
            />
          </Field>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <Field
            id={fieldId("model-provider")}
            label="Model Provider"
            hint={
              compatibilityRequired
                ? "This provider needs explicit endpoint credentials."
                : "Official providers can keep endpoint settings empty."
            }
          >
            <select
              id={fieldId("model-provider")}
              value={form.modelProvider}
              onChange={(event) => {
                const nextProvider = event.target.value;
                const nextCapability = capabilities?.providers.find(
                  (provider) => provider.value === nextProvider,
                );
                setField("modelProvider", nextProvider);
                if (
                  !form.modelName.trim() ||
                  form.modelName === providerCapability?.default_model_name
                ) {
                  setField(
                    "modelName",
                    nextCapability?.default_model_name ?? form.modelName,
                  );
                }
              }}
              className="ui-input"
            >
              {(capabilities?.providers ?? []).map((provider) => (
                <option key={provider.value} value={provider.value}>
                  {provider.label}
                </option>
              ))}
            </select>
          </Field>

          <Field
            id={fieldId("model-name")}
            label="Model Name"
            hint="Use the provider default unless you have a reason to override it."
          >
            <input
              id={fieldId("model-name")}
              type="text"
              value={form.modelName}
              onChange={(event) => setField("modelName", event.target.value)}
              className="ui-input"
              placeholder="deepseek-chat"
            />
          </Field>
        </div>

        <Field
          id={fieldId("system-prompt")}
          label="System Prompt"
          hint="Defines the agent’s role, tone, and operating constraints."
        >
          <textarea
            id={fieldId("system-prompt")}
            value={form.systemPrompt}
            onChange={(event) => setField("systemPrompt", event.target.value)}
            rows={6}
            className="ui-input ui-textarea"
            placeholder="You are a helpful operations assistant..."
          />
        </Field>
      </section>

      <DisclosureSection
        title="Compatibility endpoint"
        description="Only open this when the provider needs a custom base URL or custom API key environment variable."
        open={openCompatibilitySection}
      >
        <div className="grid gap-4 md:grid-cols-2">
          <Field
            id={fieldId("base-url")}
            label="Base URL"
            hint="Required for explicit compatible providers."
          >
            <input
              id={fieldId("base-url")}
              type="text"
              value={form.baseUrl}
              onChange={(event) => setField("baseUrl", event.target.value)}
              className="ui-input"
              placeholder="https://api.example.com/v1"
            />
          </Field>

          <Field
            id={fieldId("api-key-env-name")}
            label="API Key Env Name"
            hint="The environment variable that stores the provider key."
          >
            <input
              id={fieldId("api-key-env-name")}
              type="text"
              value={form.apiKeyEnvName}
              onChange={(event) => setField("apiKeyEnvName", event.target.value)}
              className="ui-input"
              placeholder="MINIMAX_API_KEY"
            />
          </Field>
        </div>
      </DisclosureSection>

      <section className="space-y-5">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="ui-section-kicker">Tools</p>
              <p className="ui-section-copy">
                Start with only the capabilities this agent actually needs.
              </p>
            </div>
            <PillBadge variant="default">{form.selectedTools.length} selected</PillBadge>
          </div>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-foreground">Builtin tools</h3>
            <div className="flex flex-wrap gap-2">
              {builtinTools.length === 0 ? (
                <span className="text-sm text-ink-faint">No builtin tools available</span>
              ) : (
                builtinTools.map((tool) => (
                  <ToolToggle
                    key={tool.name}
                    tool={tool}
                    selected={form.selectedTools.includes(tool.name)}
                    onToggle={() => toggleTool(tool.name)}
                  />
                ))
              )}
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="text-sm font-medium text-foreground">Agent tools</h3>
            <div className="flex flex-wrap gap-2">
              {agentTools.length === 0 ? (
                <span className="text-sm text-ink-faint">
                  No other agents available as tools
                </span>
              ) : (
                agentTools.map((tool) => (
                  <ToolToggle
                    key={tool.name}
                    tool={tool}
                    selected={form.selectedTools.includes(tool.name)}
                    onToggle={() => toggleTool(tool.name)}
                  />
                ))
              )}
            </div>
          </div>
        </div>
      </section>

      {availableSkills.length > 0 && (
        <section className="space-y-5">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="ui-section-kicker">Skills</p>
                <p className="ui-section-copy">
                  Select which globally discovered skills this agent may use.
                </p>
              </div>
              <PillBadge variant="default">{form.selectedSkills.length} selected</PillBadge>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {availableSkills.map((skill) => (
              <button
                key={skill.name}
                type="button"
                title={skill.description}
                aria-pressed={form.selectedSkills.includes(skill.name)}
                data-selected={form.selectedSkills.includes(skill.name)}
                onClick={() => toggleSkill(skill.name)}
                className="ui-chip-toggle"
              >
                <span className="font-medium">{skill.name}</span>
                <span className="ui-chip-toggle__meta">Skill</span>
              </button>
            ))}
          </div>
        </section>
      )}

      <DisclosureSection
        title="Workflow helpers"
        description="Summaries and context compacting live here so they don't crowd the main setup."
        open={openWorkflowSection}
      >
        <ToggleCard
          id={fieldId("enable-termination-summary")}
          label="Enable Termination Summary"
          description="Generate a short summary when the run settles."
          checked={form.enableTerminationSummary}
          onChange={(checked) => setField("enableTerminationSummary", checked)}
        />

        {form.enableTerminationSummary && (
          <Field
            id={fieldId("termination-summary-prompt")}
            label="Termination Summary Prompt"
            hint="Optional custom summary prompt. Leave empty to use the default summary behavior."
          >
            <textarea
              id={fieldId("termination-summary-prompt")}
              value={form.terminationSummaryPrompt}
              onChange={(event) =>
                setField("terminationSummaryPrompt", event.target.value)
              }
              rows={3}
              className="ui-input ui-textarea"
              placeholder="Optional custom prompt for the final summary"
            />
          </Field>
        )}

        <Field
          id={fieldId("compact-prompt")}
          label="Compact Prompt"
          hint="Optional prompt used when the runtime compacts long context."
        >
          <textarea
            id={fieldId("compact-prompt")}
            value={form.compactPrompt}
            onChange={(event) => setField("compactPrompt", event.target.value)}
            rows={4}
            className="ui-input ui-textarea"
            placeholder="Optional compact prompt"
          />
        </Field>

        <div className="grid gap-4 md:grid-cols-2">
          <ToggleCard
            id={fieldId("enable-context-rollback")}
            label="Enable Context Rollback"
            description="Drop no-progress periodic rounds so long-running loops do not keep useless context."
            checked={form.enableContextRollback}
            onChange={(checked) => setField("enableContextRollback", checked)}
          />

          <ToggleCard
            id={fieldId("enable-tool-retrospect")}
            label="Enable Tool Retrospect"
            description="Condense bulky tool outputs before they start dominating the prompt window."
            checked={form.enableToolRetrospect}
            onChange={(checked) => setField("enableToolRetrospect", checked)}
          />
        </div>

        {form.enableToolRetrospect && (
          <div className="grid gap-4 md:grid-cols-3">
            <Field
              id={fieldId("retrospect-token-threshold")}
              label="Retrospect Token Threshold"
              hint="Retrospect once a single tool result exceeds this token estimate."
            >
              <input
                id={fieldId("retrospect-token-threshold")}
                type="number"
                value={form.retrospectTokenThreshold}
                onChange={(event) =>
                  setBoundedIntegerField(
                    "retrospectTokenThreshold",
                    event.target.value,
                    1,
                    131072,
                  )
                }
                min={1}
                max={131072}
                className="ui-input"
              />
            </Field>

            <Field
              id={fieldId("retrospect-round-interval")}
              label="Retrospect Round Interval"
              hint="Force a retrospect pass every N rounds when tool usage keeps growing."
            >
              <input
                id={fieldId("retrospect-round-interval")}
                type="number"
                value={form.retrospectRoundInterval}
                onChange={(event) =>
                  setBoundedIntegerField(
                    "retrospectRoundInterval",
                    event.target.value,
                    1,
                    100,
                  )
                }
                min={1}
                max={100}
                className="ui-input"
              />
            </Field>

            <Field
              id={fieldId("retrospect-accumulated-token-threshold")}
              label="Retrospect Accumulated Token Threshold"
              hint="Retrospect once multiple tool results together exceed this token estimate."
            >
              <input
                id={fieldId("retrospect-accumulated-token-threshold")}
                type="number"
                value={form.retrospectAccumulatedTokenThreshold}
                onChange={(event) =>
                  setBoundedIntegerField(
                    "retrospectAccumulatedTokenThreshold",
                    event.target.value,
                    1,
                    262144,
                  )
                }
                min={1}
                max={262144}
                className="ui-input"
              />
            </Field>
          </div>
        )}
      </DisclosureSection>

      <DisclosureSection
        title="Runtime limits"
        description="Resource and orchestration limits. Leave these alone unless you need tighter operational guardrails."
      >
        <div className="grid gap-4 md:grid-cols-2">
          <Field
            id={fieldId("config-root")}
            label="Config Root"
            hint="Optional workspace root override."
          >
            <input
              id={fieldId("config-root")}
              type="text"
              value={form.configRoot}
              onChange={(event) => setField("configRoot", event.target.value)}
              className="ui-input"
              placeholder="Optional override for workspace root"
            />
          </Field>

          <Field
            id={fieldId("max-steps")}
            label="Max Steps"
            hint="Stops runaway runs before they become expensive."
          >
            <input
              id={fieldId("max-steps")}
              type="number"
              value={form.maxSteps}
              onChange={(event) => setField("maxSteps", Number(event.target.value))}
              min={1}
              max={100}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("run-timeout")}
            label="Run Timeout (s)"
            hint="How long a single run can stay active before timing out."
          >
            <input
              id={fieldId("run-timeout")}
              type="number"
              value={form.runTimeout}
              onChange={(event) => setField("runTimeout", Number(event.target.value))}
              min={10}
              max={3600}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("max-input-per-call")}
            label="Max Input Tokens Per Call"
            hint="Leave empty to derive from the context and output limits."
          >
            <input
              id={fieldId("max-input-per-call")}
              type="number"
              value={form.maxInputTokensPerCall}
              onChange={(event) =>
                setField("maxInputTokensPerCall", event.target.value)
              }
              min={1}
              max={512000}
              className="ui-input"
              placeholder="Auto"
            />
          </Field>

          <Field
            id={fieldId("max-run-cost")}
            label="Max Run Cost (USD)"
            hint="Optional hard ceiling for token cost."
          >
            <input
              id={fieldId("max-run-cost")}
              type="number"
              value={form.maxRunCost}
              onChange={(event) => setField("maxRunCost", event.target.value)}
              min={0}
              step={0.000001}
              className="ui-input"
              placeholder="Optional"
            />
          </Field>

          <Field
            id={fieldId("relevant-memory-max-token")}
            label="Relevant Memory Max Token"
            hint="Upper bound for relevant memory retrieval."
          >
            <input
              id={fieldId("relevant-memory-max-token")}
              type="number"
              value={form.relevantMemoryMaxToken}
              onChange={(event) =>
                setField("relevantMemoryMaxToken", Number(event.target.value))
              }
              min={1}
              max={32768}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("stream-cleanup-timeout")}
            label="Stream Cleanup Timeout (s)"
            hint="How long the runtime keeps stream cleanup watchers alive."
          >
            <input
              id={fieldId("stream-cleanup-timeout")}
              type="number"
              value={form.streamCleanupTimeout}
              onChange={(event) =>
                setField("streamCleanupTimeout", Number(event.target.value))
              }
              min={1}
              step={0.1}
              className="ui-input"
            />
          </Field>
        </div>
      </DisclosureSection>

      <DisclosureSection
        title="Model tuning"
        description="Sampling, context, and pricing overrides. Keep the defaults until you have a concrete reason to tune them."
      >
        <div className="grid gap-4 md:grid-cols-2">
          <Field
            id={fieldId("max-output-tokens")}
            label="Max Output Tokens"
            hint="Upper bound for each model response."
          >
            <input
              id={fieldId("max-output-tokens")}
              type="number"
              value={form.maxOutputTokens}
              onChange={(event) =>
                setField("maxOutputTokens", Number(event.target.value))
              }
              min={1}
              max={128000}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("max-context-window")}
            label="Max Context Window"
            hint="Provider-specific context limit."
          >
            <input
              id={fieldId("max-context-window")}
              type="number"
              value={form.maxContextWindow}
              onChange={(event) =>
                setField("maxContextWindow", Number(event.target.value))
              }
              min={1}
              max={512000}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("temperature")}
            label="Temperature"
            hint="Higher values produce more varied output."
          >
            <input
              id={fieldId("temperature")}
              type="number"
              value={form.temperature}
              onChange={(event) => setField("temperature", Number(event.target.value))}
              min={0}
              max={2}
              step={0.1}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("top-p")}
            label="Top P"
            hint="Alternative sampling control. Usually left at 1."
          >
            <input
              id={fieldId("top-p")}
              type="number"
              value={form.topP}
              onChange={(event) => setField("topP", Number(event.target.value))}
              min={0}
              max={1}
              step={0.01}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("frequency-penalty")}
            label="Frequency Penalty"
            hint="Discourages repeated tokens."
          >
            <input
              id={fieldId("frequency-penalty")}
              type="number"
              value={form.frequencyPenalty}
              onChange={(event) =>
                setField("frequencyPenalty", Number(event.target.value))
              }
              min={-2}
              max={2}
              step={0.1}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("presence-penalty")}
            label="Presence Penalty"
            hint="Encourages the model to introduce new topics."
          >
            <input
              id={fieldId("presence-penalty")}
              type="number"
              value={form.presencePenalty}
              onChange={(event) =>
                setField("presencePenalty", Number(event.target.value))
              }
              min={-2}
              max={2}
              step={0.1}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("cache-hit-price")}
            label="Cache-Hit Price (USD / 1M Tokens)"
            hint="Optional pricing override for cache hits."
          >
            <input
              id={fieldId("cache-hit-price")}
              type="number"
              value={form.cacheHitPrice}
              onChange={(event) =>
                setField("cacheHitPrice", Number(event.target.value))
              }
              min={0}
              step={0.000001}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("input-price")}
            label="Input Price (USD / 1M Tokens)"
            hint="Optional pricing override for prompt tokens."
          >
            <input
              id={fieldId("input-price")}
              type="number"
              value={form.inputPrice}
              onChange={(event) => setField("inputPrice", Number(event.target.value))}
              min={0}
              step={0.000001}
              className="ui-input"
            />
          </Field>

          <Field
            id={fieldId("output-price")}
            label="Output Price (USD / 1M Tokens)"
            hint="Optional pricing override for completion tokens."
          >
            <input
              id={fieldId("output-price")}
              type="number"
              value={form.outputPrice}
              onChange={(event) => setField("outputPrice", Number(event.target.value))}
              min={0}
              step={0.000001}
              className="ui-input"
            />
          </Field>
        </div>
      </DisclosureSection>

      <div className="flex flex-wrap gap-3 pt-2">
        <button type="submit" disabled={submitting} className="ui-button ui-button-primary">
          {submitting ? "Saving..." : submitLabel}
        </button>
        <Link href="/agents" className={cn("ui-button ui-button-secondary")}>
          Cancel
        </Link>
      </div>
    </form>
  );
}
