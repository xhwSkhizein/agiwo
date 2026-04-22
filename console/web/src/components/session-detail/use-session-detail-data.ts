"use client";

import { useEffect, useState } from "react";

import {
  getSessionDetail,
  getSessionSteps,
  listRuns,
} from "@/lib/api";
import type { RunResponse, SessionDetail, StepResponse } from "@/lib/api";

type DetailState = {
  key: string | null;
  detail: SessionDetail | null;
  error: string | null;
};

export function useSessionDetailResource(sessionId: string) {
  const [state, setState] = useState<DetailState>({
    key: null,
    detail: null,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    getSessionDetail(sessionId)
      .then((nextDetail) => {
        if (cancelled) {
          return;
        }
        setState({
          key: sessionId,
          detail: nextDetail,
          error: null,
        });
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setState({
          key: sessionId,
          detail: null,
          error: err instanceof Error ? err.message : "Failed to load session",
        });
      });

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  return {
    detail: state.detail,
    loading: state.key !== sessionId,
    error: state.error,
  };
}

type RunsState = {
  key: string | null;
  runs: RunResponse[];
  hasMore: boolean;
  total: number | null;
  error: string | null;
};

export function useSessionRunsPage(
  sessionId: string,
  pageSize: number,
  offset: number,
) {
  const requestKey = `${sessionId}:${pageSize}:${offset}`;
  const [state, setState] = useState<RunsState>({
    key: null,
    runs: [],
    hasMore: false,
    total: null,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    listRuns({ session_id: sessionId, limit: pageSize, offset })
      .then((page) => {
        if (cancelled) {
          return;
        }
        setState({
          key: requestKey,
          runs: page.items,
          hasMore: page.has_more,
          total: page.total,
          error: null,
        });
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setState({
          key: requestKey,
          runs: [],
          hasMore: false,
          total: null,
          error: err instanceof Error ? err.message : "Failed to load runs",
        });
      });

    return () => {
      cancelled = true;
    };
  }, [offset, pageSize, requestKey, sessionId]);

  return {
    runs: state.runs,
    loading: state.key !== requestKey,
    error: state.error,
    hasMore: state.hasMore,
    total: state.total,
  };
}

type StepsState = {
  key: string | null;
  steps: StepResponse[];
  hasMore: boolean;
  error: string | null;
};

export function useSessionStepsFeed(sessionId: string) {
  const [state, setState] = useState<StepsState>({
    key: null,
    steps: [],
    hasMore: false,
    error: null,
  });
  const [loadingMore, setLoadingMore] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getSessionSteps(sessionId, { limit: 100, order: "desc" })
      .then((page) => {
        if (cancelled) {
          return;
        }
        setState({
          key: sessionId,
          steps: [...page.items].reverse(),
          hasMore: page.has_more,
          error: null,
        });
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setState({
          key: sessionId,
          steps: [],
          hasMore: false,
          error: err instanceof Error ? err.message : "Failed to load steps",
        });
      });

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  async function loadEarlier() {
    if (state.steps.length === 0 || loadingMore) {
      return;
    }
    const oldestSequence = state.steps[0]?.sequence;
    if (!oldestSequence || oldestSequence <= 1) {
      setState((current) => ({ ...current, hasMore: false }));
      return;
    }
    setLoadingMore(true);
    try {
      const nextPage = await getSessionSteps(sessionId, {
        limit: 100,
        order: "desc",
        end_seq: oldestSequence - 1,
      });
      setState((current) => ({
        ...current,
        steps: [...nextPage.items.reverse(), ...current.steps],
        hasMore: nextPage.has_more,
      }));
    } catch (err) {
      setState((current) => ({
        ...current,
        error: err instanceof Error ? err.message : "Failed to load older steps",
      }));
    } finally {
      setLoadingMore(false);
    }
  }

  return {
    steps: state.steps,
    loading: state.key !== sessionId,
    error: state.error,
    hasMore: state.hasMore,
    loadingMore,
    loadEarlier,
  };
}
