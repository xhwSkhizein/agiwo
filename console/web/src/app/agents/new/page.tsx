"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

import { AgentForm } from "@/components/agent-form";
import { createAgent, AgentConfigCreate } from "@/lib/api";
import { useState } from "react";

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
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Link
          href="/agents"
          className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <h1 className="text-xl font-semibold">Create Agent</h1>
      </div>

      <AgentForm
        submitLabel="Create Agent"
        submitting={saving}
        error={error}
        onSubmit={handleSubmit}
      />
    </div>
  );
}
