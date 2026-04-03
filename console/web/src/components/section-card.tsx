"use client";

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type SectionCardProps = {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  headerClassName?: string;
  titleClassName?: string;
  bodyClassName?: string;
  hoverable?: boolean;
};

/**
 * Renders a reusable card with an optional header and customizable class overrides.
 *
 * @param props - Component props.
 * @param props.title - Optional header title content.
 * @param props.action - Optional header action element displayed to the right of the title.
 * @param props.children - Card content.
 * @param props.className - Additional classes merged onto the root container.
 * @param props.headerClassName - Classes merged onto the header container.
 * @param props.titleClassName - Classes merged onto the title element.
 * @param props.bodyClassName - If provided, wraps `children` in a `<div>` with these classes; otherwise `children` are rendered directly.
 * @param props.hoverable - When `true`, enables hover styles on the root container (default `false`).
 * @returns The rendered section card React element.
 */
export function SectionCard({
  title,
  action,
  children,
  className,
  headerClassName,
  titleClassName,
  bodyClassName,
  hoverable = false,
}: SectionCardProps) {
  const hasHeader = title !== undefined || action !== undefined;

  return (
    <div
      className={cn(
        "overflow-hidden rounded-2xl border border-line bg-panel shadow-sm",
        hoverable && "transition-all duration-200 hover:border-line-strong hover:bg-panel-strong",
        className
      )}
    >
      {hasHeader && (
        <div
          className={cn(
            "flex items-center justify-between border-b border-line bg-panel-muted px-4 py-3",
            headerClassName
          )}
        >
          <div className={cn("text-sm font-medium text-foreground", titleClassName)}>{title}</div>
          {action}
        </div>
      )}
      {bodyClassName ? <div className={bodyClassName}>{children}</div> : children}
    </div>
  );
}

export default SectionCard;
