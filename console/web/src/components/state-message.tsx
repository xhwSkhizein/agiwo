"use client";

import type { ReactNode } from "react";
import { AlertCircle, Info, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

type TextStateMessageProps = {
  children: ReactNode;
  className?: string;
  icon?: ReactNode;
};

export function TextStateMessage({
  children,
  className,
  icon,
}: TextStateMessageProps) {
  return (
    <div role="status" aria-live="polite" className={cn("flex items-center gap-2 text-ink-muted", className)}>
      {icon}
      {children}
    </div>
  );
}

export function EmptyStateMessage({
  children,
  className,
}: Omit<TextStateMessageProps, "icon">) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn("flex flex-col items-center justify-center gap-3 py-12 text-center", className)}
    >
      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-panel-strong">
        <Info className="h-5 w-5 text-ink-muted" />
      </div>
      <div className="text-sm text-ink-muted">{children}</div>
    </div>
  );
}

export function FullPageMessage({
  children,
  className,
  loading = false,
}: TextStateMessageProps & { loading?: boolean }) {
  return (
    <div className="flex items-center justify-center h-full">
      <div role="status" aria-live="polite" className={cn("flex items-center gap-2 text-ink-muted", className)}>
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
        {children}
      </div>
    </div>
  );
}

export function ErrorStateMessage({
  children,
  className,
}: Omit<TextStateMessageProps, "icon">) {
  return (
    <div
      role="alert"
      className={cn(
        "rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3",
        "flex items-start gap-3"
      )}
    >
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-danger" />
      <div className={cn("text-sm text-danger", className)}>{children}</div>
    </div>
  );
}

export default FullPageMessage;
