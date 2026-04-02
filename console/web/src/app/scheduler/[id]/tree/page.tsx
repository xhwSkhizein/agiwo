"use client";

import { useCallback } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";

import { SchedulerTreeWorkspace } from "@/components/scheduler-tree/scheduler-tree-workspace";

export default function SchedulerTreePage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();

  const rootStateId = params.id as string;
  const selectedStateId = searchParams.get("selected");

  const handleSelectedStateIdChange = useCallback(
    (stateId: string) => {
      const nextQuery = new URLSearchParams(searchParams.toString());
      if (stateId === rootStateId) {
        nextQuery.delete("selected");
      } else {
        nextQuery.set("selected", stateId);
      }
      const suffix = nextQuery.toString();
      router.replace(
        suffix
          ? `/scheduler/${rootStateId}/tree?${suffix}`
          : `/scheduler/${rootStateId}/tree`,
      );
    },
    [rootStateId, router, searchParams],
  );

  return (
    <SchedulerTreeWorkspace
      rootStateId={rootStateId}
      selectedStateId={selectedStateId}
      onSelectedStateIdChange={handleSelectedStateIdChange}
    />
  );
}
