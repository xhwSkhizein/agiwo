"use client";

import type { ReactNode } from "react";

type SectionCardProps = {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  headerClassName?: string;
  titleClassName?: string;
  bodyClassName?: string;
};

export function SectionCard({
  title,
  action,
  children,
  className = "",
  headerClassName = "px-4 py-3 border-b border-zinc-800 flex items-center justify-between",
  titleClassName = "text-sm font-medium",
  bodyClassName,
}: SectionCardProps) {
  const hasHeader = title !== undefined || action !== undefined;

  return (
    <div className={`rounded-lg border border-zinc-800 bg-zinc-900 ${className}`}>
      {hasHeader && (
        <div className={headerClassName}>
          <div className={titleClassName}>{title}</div>
          {action}
        </div>
      )}
      {bodyClassName ? <div className={bodyClassName}>{children}</div> : children}
    </div>
  );
}

export default SectionCard;
