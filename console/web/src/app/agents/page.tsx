"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Plus, Trash2, Network, Settings } from "lucide-react";
import { ErrorStateMessage, EmptyStateMessage, FullPageMessage } from "@/components/state-message";
import { PillBadge } from "@/components/pill-badge";
import { MonoText } from "@/components/mono-text";
import { listAgents, deleteAgent } from "@/lib/api";
import type { AgentConfig } from "@/lib/api";

/**
 * Render the Agents management page with listing, creation, and deletion controls.
 *
 * Loads agent configurations on mount and displays loading, empty, and error states.
 * Provides UI actions to create a new agent, open an agent's scheduler or configuration,
 * and delete agents via an inline confirmation flow (with per-item pending/error state).
 *
 * @returns The rendered Agents page as a JSX element
 */
export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const [deletePendingId, setDeletePendingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const fetchAgents = () =>
    listAgents()
      .then((items) => {
        setAgents(items);
        setLoadError(null);
      })
      .catch((err) => {
        setAgents([]);
        setLoadError(err instanceof Error ? err.message : "Failed to load agents");
      });

  const loadAgents = (showLoading = true) => {
    if (showLoading) {
      setLoading(true);
    }
    fetchAgents().finally(() => setLoading(false));
  };

  useEffect(() => {
    void fetchAgents().finally(() => setLoading(false));
  }, []);

  const handleDelete = async (id: string, name: string) => {
    setDeletePendingId(id);
    setDeleteError(null);
    try {
      await deleteAgent(id);
      setConfirmingDeleteId(null);
      loadAgents();
    } catch (err) {
      setDeleteError(
        err instanceof Error ? err.message : `Failed to delete agent "${name}"`,
      );
    } finally {
      setDeletePendingId(null);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Agents</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Create and manage agent configurations
          </p>
        </div>
        <Link
          href="/agents/new"
          className="ui-button ui-button-primary"
        >
          <Plus className="w-4 h-4" />
          New Agent
        </Link>
      </div>

      {loadError && <ErrorStateMessage>{loadError}</ErrorStateMessage>}
      {deleteError && <ErrorStateMessage>{deleteError}</ErrorStateMessage>}

      {loading ? (
        <FullPageMessage loading>Loading agents...</FullPageMessage>
      ) : agents.length === 0 ? (
        <EmptyStateMessage>
          <div className="space-y-4">
            <p>No agents configured yet</p>
            <Link
              href="/agents/new"
              className="ui-button ui-button-secondary"
            >
              <Plus className="w-4 h-4" />
              Create your first agent
            </Link>
          </div>
        </EmptyStateMessage>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((agent) => (
            <div
              key={agent.id}
              className="flex flex-col rounded-2xl border border-line bg-panel p-5"
            >
              <div className="flex items-start justify-between">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="font-medium text-foreground">{agent.name}</h3>
                    {agent.is_default && (
                      <PillBadge variant="success">Default</PillBadge>
                    )}
                  </div>
                  <MonoText className="mt-0.5">{agent.model_provider}/{agent.model_name}</MonoText>
                </div>
                {!agent.is_default && (
                  <button
                    type="button"
                    onClick={() => {
                      setDeleteError(null);
                      setConfirmingDeleteId((current) =>
                        current === agent.id ? null : agent.id,
                      );
                    }}
                    aria-label={`Delete ${agent.name}`}
                    aria-pressed={confirmingDeleteId === agent.id}
                    className="ui-button ui-button-ghost ui-button-icon"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              {agent.description && (
                <p className="mt-2 line-clamp-2 text-sm text-ink-muted">
                  {agent.description}
                </p>
              )}

              {agent.tools && agent.tools.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-3">
                  {agent.tools.slice(0, 6).map((t) => (
                    <PillBadge
                      key={t}
                      variant={t.startsWith("agent:") ? "info" : "default"}
                    >
                      {t.startsWith("agent:") ? t.slice(6) : t}
                    </PillBadge>
                  ))}
                  {agent.tools.length > 6 && (
                    <PillBadge variant="default">+{agent.tools.length - 6}</PillBadge>
                  )}
                </div>
              )}

              {confirmingDeleteId === agent.id && (
                <div className="ui-inline-confirm mt-4 space-y-3 p-3">
                  <p className="text-sm text-red-200">
                    Delete <span className="font-medium">{agent.name}</span>? This cannot be
                    undone.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => handleDelete(agent.id, agent.name)}
                      disabled={deletePendingId === agent.id}
                      className="ui-button ui-button-danger"
                    >
                      {deletePendingId === agent.id ? "Deleting..." : "Delete agent"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmingDeleteId(null)}
                      disabled={deletePendingId === agent.id}
                      className="ui-button ui-button-secondary"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              <div className="mt-auto pt-4 grid grid-cols-2 gap-2">
                <Link
                  href={`/agents/${agent.id}/scheduler-chat`}
                  className="ui-button ui-button-primary col-span-2"
                >
                  <Network className="w-3.5 h-3.5" />
                  Scheduler
                </Link>
                <Link
                  href={`/agents/${agent.id}/edit`}
                  className="ui-button ui-button-secondary col-span-2"
                >
                  <Settings className="w-3.5 h-3.5" />
                  Configure
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
