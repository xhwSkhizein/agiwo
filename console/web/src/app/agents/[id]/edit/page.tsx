"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { AgentForm } from "@/components/agent-form";
import { BackHeader } from "@/components/back-header";
import { FullPageMessage } from "@/components/state-message";
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
    return <FullPageMessage>Loading...</FullPageMessage>;
  }

  if (agent === null) {
    return (
      <FullPageMessage className="text-red-400">
        {error ?? "Agent not found"}
      </FullPageMessage>
    );
  }

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <BackHeader
        href="/agents"
        title="Edit Agent"
        subtitle={null}
      />

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
