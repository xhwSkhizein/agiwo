"use client";

import { useRouter } from "next/navigation";

import { AgentForm } from "@/components/agent-form";
import { BackHeader } from "@/components/back-header";
import { createAgent, AgentConfigCreate } from "@/lib/api";
import { useState } from "react";

/**
 * Render the Create Agent page and manage form submission state.
 *
 * Handles creating an agent using the API client, navigates to `/agents` on success,
 * exposes submission error messages on failure, and disables the form while saving.
 *
 * @returns The JSX element for the Create Agent page
 */
export default function NewAgentPage() {
  const router = useRouter();
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (payload: AgentConfigCreate) => {
    setSaving(true);
    setError(null);
    try {
      await createAgent(payload);
      router.push("/agents");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create agent");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <BackHeader
        href="/agents"
        title="Create Agent"
        subtitle={null}
      />

      <AgentForm
        submitLabel="Create Agent"
        submitting={saving}
        error={error}
        onSubmit={handleSubmit}
      />
    </div>
  );
}
