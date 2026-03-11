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
};

export function BackHeader({
  href,
  title,
  subtitle,
  rightContent,
  className = "flex items-center gap-3",
  titleClassName = "text-xl font-semibold",
  subtitleClassName = "text-xs text-zinc-500 font-mono mt-0.5",
}: BackHeaderProps) {
  return (
    <div className={className}>
      <Link
        href={href}
        className="p-1.5 rounded hover:bg-zinc-800 transition-colors"
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
