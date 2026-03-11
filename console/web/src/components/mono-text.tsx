"use client";

import type { ReactNode } from "react";
import Link from "next/link";

type MonoTextProps = {
  children: ReactNode;
  className?: string;
};

type MonoLinkProps = MonoTextProps & {
  href: string;
};

export function MonoText({
  children,
  className = "font-mono text-xs",
}: MonoTextProps) {
  return <span className={className}>{children}</span>;
}

export function MonoLink({
  href,
  children,
  className = "text-zinc-200 hover:text-white font-mono text-xs",
}: MonoLinkProps) {
  return (
    <Link href={href} className={className}>
      {children}
    </Link>
  );
}

export default MonoText;
