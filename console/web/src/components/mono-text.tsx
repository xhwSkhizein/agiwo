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

/**
 * Render a monospaced, small muted inline text element.
 *
 * @param children - Content to render inside the span.
 * @param className - Additional CSS classes to apply to the span.
 * @param truncate - If true, applies truncation styling to shorten overflowed text.
 * @param title - HTML title attribute to show on hover.
 * @returns A styled <span> element containing `children`.
 */
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

/**
 * Renders a styled monospaced link with optional external behavior and optional truncation.
 *
 * @param external - If `true`, adds `target="_blank"` and `rel="noopener noreferrer"` so the link opens in a new tab safely.
 * @param truncate - If `true`, applies a `truncate` class to ellipsize overflowing text.
 * @param title - Value forwarded to the link's `title` attribute.
 *
 * @returns A Next.js `Link` element styled as small monospaced text.
 */
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

/**
 * Renders an inline, styled code block with monospaced small text and a bordered background.
 *
 * @param children - Content to display inside the code block
 * @param className - Additional CSS classes to merge with the component's default styling
 * @returns The rendered `<code>` element containing `children`
 */
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
