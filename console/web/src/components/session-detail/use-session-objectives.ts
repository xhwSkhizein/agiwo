"use client";

import { useCallback, useEffect, useState } from "react";

import { listSessionObjectives, type ObjectiveView } from "@/lib/api";
import { pickActiveObjective } from "@/lib/objective-status";

export function useSessionObjectives(sessionId: string) {
  const [objectives, setObjectives] = useState<ObjectiveView[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const items = await listSessionObjectives(sessionId);
      setObjectives(items);
    } catch (err) {
      setObjectives([]);
      setError(err instanceof Error ? err.message : "Failed to load objectives");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  return {
    objectives,
    active: pickActiveObjective(objectives),
    loading,
    error,
    reload,
  };
}
