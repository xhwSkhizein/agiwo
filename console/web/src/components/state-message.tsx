"use client";

import type { ReactNode } from "react";
import { AlertCircle, Info, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

type TextStateMessageProps = {
  children: ReactNode;
  className?: string;
  icon?: ReactNode;
};

/**
 * Render a compact, accessible inline status message with an optional leading icon.
 *
 * Renders a container with role="status" and aria-live="polite" to announce its content to assistive technologies.
 *
 * @param children - Message content to display inside the status container
 * @param className - Additional CSS class names to merge with the component's base classes
 * @param icon - Optional leading icon or element displayed before the message content
 * @returns A JSX element representing the styled status message container
 */
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

/**
 * Displays a centered empty-state message with an info icon.
 *
 * Renders a vertically stacked, centered container showing an information icon inside a circular background and the provided message text in muted, small styling.
 *
 * @param children - Content to display as the empty-state message
 * @param className - Additional CSS classes to apply to the outer container
 * @returns A React element representing the empty-state message
 */
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

/**
 * Render a centered full-page status container that optionally displays a loading spinner.
 *
 * @param className - Additional CSS classes to merge into the inner status container
 * @param loading - If `true`, renders a spinning loader icon before the children
 * @returns A JSX element containing a centered status container with an optional spinner and the provided children
 */
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

/**
 * Render a bordered error message container with an alert icon and danger styling.
 *
 * Renders a rounded, colored container with an alert icon on the left and the provided content on the right. The container uses `role="alert"` for accessibility and applies danger-themed styles; additional CSS classes for the text wrapper can be supplied.
 *
 * @param children - Content to display as the error message
 * @param className - Additional CSS classes to apply to the text wrapper
 * @returns A JSX element containing the styled error message with an alert icon
 */
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
