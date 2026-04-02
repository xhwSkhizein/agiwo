"use client";

import type { ReactNode } from "react";

type TextStateMessageProps = {
  children: ReactNode;
  className?: string;
};

export function TextStateMessage({
  children,
  className = "text-zinc-500",
}: TextStateMessageProps) {
  return <div className={className}>{children}</div>;
}

export function EmptyStateMessage({
  children,
  className = "text-zinc-500 text-center py-12",
}: TextStateMessageProps) {
  return <div className={className}>{children}</div>;
}

export function FullPageMessage({
  children,
  className = "text-zinc-500",
}: TextStateMessageProps) {
  return (
    <div className="flex items-center justify-center h-full">
      <div className={className}>{children}</div>
    </div>
  );
}

export function ErrorStateMessage({
  children,
  className = "rounded-md border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200",
}: TextStateMessageProps) {
  return <div className={className}>{children}</div>;
}

export default FullPageMessage;
