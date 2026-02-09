"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { getAgent, updateAgent, listAvailableTools, AvailableTool } from "@/lib/api";

const PROVIDERS = [
  { value: "openai", label: "OpenAI" },
  { value: "deepseek", label: "DeepSeek" },
  { value: "anthropic", label: "Anthropic" },
];

export default function EditAgentPage() {
  const params = useParams();
  const agentId = params.id as string;
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [modelProvider, setModelProvider] = useState("deepseek");
  const [modelName, setModelName] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [maxSteps, setMaxSteps] = useState(10);
  const [runTimeout, setRunTimeout] = useState(600);
  const [maxOutputTokens, setMaxOutputTokens] = useState(8196);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [temperature, setTemperature] = useState(0.7);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [availableTools, setAvailableTools] = useState<AvailableTool[]>([]);
  const [enableSkill, setEnableSkill] = useState(false);
  const [skillsDir, setSkillsDir] = useState("");

  useEffect(() => {
    Promise.all([getAgent(agentId), listAvailableTools()])
      .then(([a, tools]) => {
        setName(a.name);
        setDescription(a.description);
        setModelProvider(a.model_provider);
        setModelName(a.model_name);
        setSystemPrompt(a.system_prompt);
        setMaxSteps((a.options?.max_steps as number) ?? 10);
        setRunTimeout((a.options?.run_timeout as number) ?? 600);
        setMaxOutputTokens((a.options?.max_output_tokens as number) ?? 8196);
        setMaxTokens((a.model_params?.max_tokens as number) ?? 4096);
        setTemperature((a.model_params?.temperature as number) ?? 0.7);
        setSelectedTools(a.tools || []);
        setEnableSkill((a.options?.enable_skill as boolean) ?? false);
        setSkillsDir((a.options?.skills_dir as string) ?? "");
        setAvailableTools(tools);
      })
      .catch(() => setError("Agent not found"))
      .finally(() => setLoading(false));
  }, [agentId]);

  const toggleTool = (toolName: string) => {
    setSelectedTools((prev) =>
      prev.includes(toolName) ? prev.filter((t) => t !== toolName) : [...prev, toolName]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await updateAgent(agentId, {
        name,
        description,
        model_provider: modelProvider,
        model_name: modelName,
        system_prompt: systemPrompt,
        tools: selectedTools,
        options: {
          max_steps: maxSteps,
          run_timeout: runTimeout,
          max_output_tokens: maxOutputTokens,
          enable_skill: enableSkill,
          skills_dir: skillsDir || undefined,
        },
        model_params: {
          max_tokens: maxTokens,
          temperature,
        },
      });
      router.push("/agents");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update agent");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Loading...</div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Link
          href="/agents"
          className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <h1 className="text-xl font-semibold">Edit Agent</h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>

        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">Description</label>
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Model Provider</label>
            <select
              value={modelProvider}
              onChange={(e) => setModelProvider(e.target.value)}
              className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            >
              {PROVIDERS.map((p) => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Model Name</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm text-zinc-400 mb-1.5">System Prompt</label>
          <textarea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={5}
            className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600 resize-y"
          />
        </div>

        <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">Tools</p>
        <div className="flex flex-wrap gap-2">
          {availableTools.map((tool) => (
            <button
              key={tool.name}
              type="button"
              onClick={() => toggleTool(tool.name)}
              className={`px-3 py-1.5 rounded-md border text-sm transition-colors ${
                selectedTools.includes(tool.name)
                  ? "bg-white text-black border-white"
                  : "bg-zinc-900 text-zinc-400 border-zinc-700 hover:border-zinc-500"
              }`}
              title={tool.description}
            >
              {tool.name}
            </button>
          ))}
          {availableTools.length === 0 && (
            <span className="text-xs text-zinc-600">No tools available</span>
          )}
        </div>

        <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">Skills</p>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-zinc-400 cursor-pointer">
            <input
              type="checkbox"
              checked={enableSkill}
              onChange={(e) => setEnableSkill(e.target.checked)}
              className="rounded border-zinc-700 bg-zinc-900"
            />
            Enable Skills
          </label>
          {enableSkill && (
            <input
              type="text"
              value={skillsDir}
              onChange={(e) => setSkillsDir(e.target.value)}
              placeholder="Skills directory (optional, uses default)"
              className="flex-1 px-3 py-1.5 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            />
          )}
        </div>

        <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">Agent Options</p>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Max Steps</label>
            <input
              type="number"
              value={maxSteps}
              onChange={(e) => setMaxSteps(Number(e.target.value))}
              min={1}
              max={100}
              className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Run Timeout (s)</label>
            <input
              type="number"
              value={runTimeout}
              onChange={(e) => setRunTimeout(Number(e.target.value))}
              min={10}
              max={3600}
              className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Max Output Tokens</label>
            <input
              type="number"
              value={maxOutputTokens}
              onChange={(e) => setMaxOutputTokens(Number(e.target.value))}
              min={256}
              max={128000}
              className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            />
          </div>
        </div>

        <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider pt-2">Model Parameters</p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Max Tokens</label>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              min={256}
              max={128000}
              className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Temperature</label>
            <input
              type="number"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              min={0}
              max={2}
              step={0.1}
              className="w-full px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 text-sm focus:outline-none focus:border-zinc-600"
            />
          </div>
        </div>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <div className="flex gap-3 pt-2">
          <button
            type="submit"
            disabled={saving}
            className="px-5 py-2 rounded-md bg-white text-black text-sm font-medium hover:bg-zinc-200 transition-colors disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save Changes"}
          </button>
          <Link
            href="/agents"
            className="px-5 py-2 rounded-md border border-zinc-700 text-sm hover:bg-zinc-800 transition-colors"
          >
            Cancel
          </Link>
        </div>
      </form>
    </div>
  );
}
