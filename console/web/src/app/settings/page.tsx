"use client";

import { useEffect, useState } from "react";

import { ErrorStateMessage, FullPageMessage } from "@/components/state-message";
import { JsonDisclosure } from "@/components/json-disclosure";
import { SectionCard } from "@/components/section-card";
import {
  AgentCapabilities,
  AvailableSkill,
  AvailableTool,
  getAgentCapabilities,
  getRuntimeConfig,
  RuntimeConfigEditable,
  RuntimeConfigSnapshot,
  listAvailableSkills,
  listAvailableTools,
  updateRuntimeConfig,
} from "@/lib/api";

type SettingsFormState = {
  skillsDirsText: string;
  defaultAgentId: string;
  defaultAgentName: string;
  defaultAgentDescription: string;
  modelProvider: string;
  modelName: string;
  systemPrompt: string;
  modelParamsJson: string;
  selectedTools: string[];
  selectedSkills: string[];
};

function buildFormState(editable: RuntimeConfigEditable): SettingsFormState {
  return {
    skillsDirsText: editable.skills_dirs.join("\n"),
    defaultAgentId: editable.default_agent.id,
    defaultAgentName: editable.default_agent.name,
    defaultAgentDescription: editable.default_agent.description,
    modelProvider: editable.default_agent.model_provider,
    modelName: editable.default_agent.model_name,
    systemPrompt: editable.default_agent.system_prompt,
    modelParamsJson: JSON.stringify(editable.default_agent.model_params, null, 2),
    selectedTools: editable.default_agent.tools,
    selectedSkills: editable.default_agent.allowed_skills,
  };
}

function normalizeLineList(value: string): string[] {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function SettingsPage() {
  const [snapshot, setSnapshot] = useState<RuntimeConfigSnapshot | null>(null);
  const [providers, setProviders] = useState<AgentCapabilities["providers"]>([]);
  const [tools, setTools] = useState<AvailableTool[]>([]);
  const [skills, setSkills] = useState<AvailableSkill[]>([]);
  const [form, setForm] = useState<SettingsFormState | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const reload = async () => {
    setLoading(true);
    try {
      const [config, capabilities, availableTools, availableSkills] = await Promise.all([
        getRuntimeConfig(),
        getAgentCapabilities(),
        listAvailableTools(),
        listAvailableSkills(),
      ]);
      setSnapshot(config);
      setProviders(capabilities.providers);
      setTools(availableTools);
      setSkills(availableSkills);
      setForm(buildFormState(config.editable));
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Failed to load runtime config");
    } finally {
      setLoading(false);
    }
  };

  const refreshSkills = async () => {
    const availableSkills = await listAvailableSkills();
    setSkills(availableSkills);
    return availableSkills;
  };

  useEffect(() => {
    void reload();
  }, []);

  const toggleSelection = (
    value: string,
    selected: string[],
    setter: (next: string[]) => void,
  ) => {
    if (selected.includes(value)) {
      setter(selected.filter((item) => item !== value));
      return;
    }
    setter([...selected, value]);
  };

  const handleSave = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!form) {
      return;
    }
    setSaving(true);
    setSaveError(null);
    setSaveMessage(null);

    try {
      const parsedModelParams = JSON.parse(
        form.modelParamsJson
      ) as RuntimeConfigEditable["default_agent"]["model_params"];
      const nextSnapshot = await updateRuntimeConfig({
        skills_dirs: normalizeLineList(form.skillsDirsText),
        default_agent: {
          id: form.defaultAgentId,
          name: form.defaultAgentName.trim(),
          description: form.defaultAgentDescription.trim(),
          model_provider: form.modelProvider,
          model_name: form.modelName.trim(),
          system_prompt: form.systemPrompt,
          tools: form.selectedTools,
          allowed_skills: form.selectedSkills,
          model_params: parsedModelParams,
        },
      });
      setSnapshot(nextSnapshot);
      const refreshedSkills = await refreshSkills();
      const allowedSkillSet = new Set(refreshedSkills.map((skill) => skill.name));
      const nextForm = buildFormState(nextSnapshot.editable);
      nextForm.selectedSkills = nextForm.selectedSkills.filter((skill) =>
        allowedSkillSet.has(skill)
      );
      setForm(nextForm);
      setSaveMessage("Runtime config updated. Changes apply to new work after this save.");
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Failed to update runtime config");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <FullPageMessage loading>Loading runtime config...</FullPageMessage>;
  }

  if (!form || !snapshot) {
    return (
      <div className="mx-auto max-w-6xl p-6">
        <ErrorStateMessage>{loadError ?? "Runtime config is unavailable"}</ErrorStateMessage>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Settings</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Inspect effective global config and update the runtime-only override for
            <code className="ml-1 rounded bg-panel-muted px-1.5 py-0.5 text-xs">skills_dirs</code>
            and the default agent.
          </p>
        </div>
        <button type="button" onClick={() => void reload()} className="ui-button ui-button-secondary">
          Refresh
        </button>
      </div>

      <SectionCard
        title="Runtime scope"
        bodyClassName="space-y-3 px-4 py-4"
      >
        <p className="text-sm text-ink-soft">
          This page only updates the running process. Restarting Console will restore the
          environment-backed config.
        </p>
        <div className="flex flex-wrap gap-2">
          {snapshot.restart_required.map((item) => (
            <span
              key={item}
              className="rounded-full border border-line px-3 py-1 text-xs text-ink-muted"
            >
              Restart required: {item}
            </span>
          ))}
        </div>
      </SectionCard>

      {loadError && <ErrorStateMessage>{loadError}</ErrorStateMessage>}
      {saveError && <ErrorStateMessage>{saveError}</ErrorStateMessage>}
      {saveMessage && (
        <div className="rounded-2xl border border-success bg-[color:var(--success-soft)] px-4 py-3 text-sm text-foreground">
          {saveMessage}
        </div>
      )}

      <form className="space-y-6" onSubmit={handleSave}>
        <SectionCard title="Editable config" bodyClassName="space-y-5 px-4 py-4">
          <div>
            <label htmlFor="skills-dirs" className="ui-field-label">
              skills_dirs
            </label>
            <textarea
              id="skills-dirs"
              value={form.skillsDirsText}
              onChange={(event) =>
                setForm({ ...form, skillsDirsText: event.target.value })
              }
              rows={4}
              className="ui-input ui-textarea"
              placeholder="examples/skills&#10;skills"
            />
            <p className="ui-field-hint">One directory per line. Relative paths are resolved from the SDK root path.</p>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label htmlFor="default-agent-id" className="ui-field-label">
                Default Agent ID
              </label>
              <input
                id="default-agent-id"
                value={form.defaultAgentId}
                onChange={(event) =>
                  setForm({ ...form, defaultAgentId: event.target.value })
                }
                className="ui-input"
              />
            </div>
            <div>
              <label htmlFor="default-agent-name" className="ui-field-label">
                Default Agent Name
              </label>
              <input
                id="default-agent-name"
                value={form.defaultAgentName}
                onChange={(event) =>
                  setForm({ ...form, defaultAgentName: event.target.value })
                }
                className="ui-input"
              />
            </div>
            <div className="md:col-span-2">
              <label htmlFor="default-agent-description" className="ui-field-label">
                Description
              </label>
              <input
                id="default-agent-description"
                value={form.defaultAgentDescription}
                onChange={(event) =>
                  setForm({ ...form, defaultAgentDescription: event.target.value })
                }
                className="ui-input"
              />
            </div>
            <div>
              <label htmlFor="default-agent-provider" className="ui-field-label">
                Model Provider
              </label>
              <select
                id="default-agent-provider"
                value={form.modelProvider}
                onChange={(event) =>
                  setForm({ ...form, modelProvider: event.target.value })
                }
                className="ui-input"
              >
                {providers.map((provider) => (
                  <option key={provider.value} value={provider.value}>
                    {provider.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="default-agent-model-name" className="ui-field-label">
                Model Name
              </label>
              <input
                id="default-agent-model-name"
                value={form.modelName}
                onChange={(event) =>
                  setForm({ ...form, modelName: event.target.value })
                }
                className="ui-input"
                placeholder={
                  providers.find((item) => item.value === form.modelProvider)?.default_model_name ??
                  ""
                }
              />
            </div>
            <div className="md:col-span-2">
              <label htmlFor="default-agent-system-prompt" className="ui-field-label">
                System Prompt
              </label>
              <textarea
                id="default-agent-system-prompt"
                value={form.systemPrompt}
                onChange={(event) =>
                  setForm({ ...form, systemPrompt: event.target.value })
                }
                rows={5}
                className="ui-input ui-textarea"
              />
            </div>
            <div className="md:col-span-2">
              <label htmlFor="default-agent-model-params" className="ui-field-label">
                Model Params JSON
              </label>
              <textarea
                id="default-agent-model-params"
                value={form.modelParamsJson}
                onChange={(event) =>
                  setForm({ ...form, modelParamsJson: event.target.value })
                }
                rows={10}
                className="ui-input ui-textarea font-mono text-xs"
              />
              <p className="ui-field-hint">
                Full model params payload. This keeps advanced provider-specific fields editable
                without opening separate controls.
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <div className="ui-field-label">Default Agent Tools</div>
              <div className="flex flex-wrap gap-2">
                {tools.map((tool) => {
                  const checked = form.selectedTools.includes(tool.name);
                  return (
                    <button
                      key={tool.name}
                      type="button"
                      aria-pressed={checked}
                      onClick={() =>
                        toggleSelection(tool.name, form.selectedTools, (next) =>
                          setForm({ ...form, selectedTools: next })
                        )
                      }
                      className={`ui-chip-toggle ${checked ? "border-accent bg-panel-strong text-foreground" : ""}`}
                    >
                      {tool.name}
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <div className="ui-field-label">Default Agent Skills</div>
              <div className="flex flex-wrap gap-2">
                {skills.map((skill) => {
                  const checked = form.selectedSkills.includes(skill.name);
                  return (
                    <button
                      key={skill.name}
                      type="button"
                      aria-pressed={checked}
                      onClick={() =>
                        toggleSelection(skill.name, form.selectedSkills, (next) =>
                          setForm({ ...form, selectedSkills: next })
                        )
                      }
                      className={`ui-chip-toggle ${checked ? "border-accent bg-panel-strong text-foreground" : ""}`}
                      title={skill.description}
                    >
                      {skill.name}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <button type="submit" disabled={saving} className="ui-button ui-button-primary">
              {saving ? "Saving..." : "Save runtime config"}
            </button>
            <button
              type="button"
              disabled={saving}
              onClick={() => setForm(buildFormState(snapshot.editable))}
              className="ui-button ui-button-secondary"
            >
              Reset form
            </button>
          </div>
        </SectionCard>
      </form>

      <SectionCard title="Effective config" bodyClassName="space-y-3 px-4 py-4">
        <JsonDisclosure label="effective" value={snapshot.effective} />
      </SectionCard>

      <SectionCard title="Read-only config" bodyClassName="space-y-3 px-4 py-4">
        {Object.entries(snapshot.readonly).map(([key, value]) => (
          <JsonDisclosure key={key} label={key} value={value} />
        ))}
      </SectionCard>
    </div>
  );
}
