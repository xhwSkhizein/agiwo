"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

import { AgentForm } from "@/components/agent-form";
import { AgentConfig, AgentConfigCreate, getAgent, updateAgent } from "@/lib/api";

export default function EditAgentPage() {
  const params = useParams();
  const agentId = params.id as string;
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agent, setAgent] = useState<AgentConfig | null>(null);

  useEffect(() => {
    getAgent(agentId)
      .then(setAgent)
      .catch(() => setError("Agent not found"))
      .finally(() => setLoading(false));
  }, [agentId]);

  const handleSubmit = async (payload: AgentConfigCreate) => {
    setSaving(true);
    setError(null);
    try {
      await updateAgent(agentId, payload);
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

  if (agent === null) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-400">{error ?? "Agent not found"}</div>
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

      <AgentForm
        initialAgent={agent}
        excludeAgentId={agentId}
        submitLabel="Save Changes"
        submitting={saving}
        error={error}
        onSubmit={handleSubmit}
      />
    </div>
  );
}
