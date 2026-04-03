"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

type BackHeaderProps = {
  href: string;
  title: ReactNode;
  subtitle?: ReactNode;
  rightContent?: ReactNode;
  className?: string;
  titleClassName?: string;
  subtitleClassName?: string;
  backLabel?: string;
};

export function BackHeader({
  href,
  title,
  subtitle,
  rightContent,
  className = "flex items-center gap-3",
  titleClassName = "text-xl font-semibold",
  subtitleClassName = "mt-0.5 font-mono text-xs text-ink-muted",
  backLabel = "Go back",
}: BackHeaderProps) {
  return (
    <div className={className}>
      <Link
        href={href}
        aria-label={backLabel}
        className="ui-button ui-button-ghost ui-button-icon"
      >
        <ArrowLeft className="w-4 h-4" />
      </Link>
      <div>
        <h1 className={titleClassName}>{title}</h1>
        {subtitle !== undefined && subtitle !== null && (
          <div className={subtitleClassName}>{subtitle}</div>
        )}
      </div>
      {rightContent && <div className="ml-auto flex items-center gap-2">{rightContent}</div>}
    </div>
  );
}

export default BackHeader;
