"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";

type MonoTextProps = {
  children: ReactNode;
  className?: string;
  truncate?: boolean;
  title?: string;
};

type MonoLinkProps = MonoTextProps & {
  href: string;
  external?: boolean;
};

export function MonoText({
  children,
  className,
  truncate = false,
  title,
}: MonoTextProps) {
  return (
    <span
      className={cn(
        "font-mono text-xs text-ink-muted",
        truncate && "truncate",
        className
      )}
      title={title}
    >
      {children}
    </span>
  );
}

export function MonoLink({
  href,
  children,
  className,
  external = false,
  truncate = false,
  title,
}: MonoLinkProps) {
  const linkProps = external
    ? { target: "_blank", rel: "noopener noreferrer" }
    : {};

  return (
    <Link
      href={href}
      className={cn(
        "font-mono text-xs text-ink-soft transition-colors duration-150 hover:text-foreground",
        "hover:underline underline-offset-2",
        truncate && "truncate",
        className
      )}
      title={title}
      {...linkProps}
    >
      {children}
    </Link>
  );
}

export function MonoBlock({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <code
      className={cn(
        "rounded border border-line bg-panel-strong px-1.5 py-0.5 font-mono text-xs",
        "text-ink-soft",
        className
      )}
    >
      {children}
    </code>
  );
}

export default MonoText;
