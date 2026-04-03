"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  MessageSquare,
  Activity,
  Bot,
  CalendarClock,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/sessions", label: "Sessions", icon: MessageSquare },
  { href: "/traces", label: "Traces", icon: Activity },
  { href: "/agents", label: "Agents", icon: Bot },
  { href: "/scheduler", label: "Scheduler", icon: CalendarClock },
];

/**
 * Renders the application's left navigation sidebar with a header, primary navigation, and footer version label.
 *
 * The primary navigation highlights the active route: the Dashboard item is active when the pathname is "/", and other items are active when the pathname starts with the item's `href`. Active links receive aria-current="page" and a visual accent.
 *
 * @returns The sidebar JSX element containing the header, primary nav, and footer version label.
 */
export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-line bg-panel">
      <div className="border-b border-line px-4 py-5">
        <h1 className="text-lg font-semibold tracking-tight text-foreground">
          Agiwo Console
        </h1>
        <p className="mt-0.5 text-xs text-ink-muted">Agent SDK Control Plane</p>
      </div>

      <nav aria-label="Primary" className="flex-1 space-y-0.5 px-2 py-3">
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                "group flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm transition-all duration-150",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
                isActive
                  ? "bg-panel-strong text-foreground font-medium"
                  : "text-ink-muted hover:bg-panel-muted hover:text-ink-soft"
              )}
            >
              <item.icon
                className={cn(
                  "w-4 h-4 transition-colors duration-150",
                  isActive ? "text-ink-soft" : "text-ink-faint group-hover:text-ink-muted"
                )}
              />
              {item.label}
              {isActive && (
                <span className="ml-auto h-1.5 w-1.5 rounded-full bg-accent" />
              )}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-line px-4 py-3 text-xs text-ink-faint">
        v0.1.0
      </div>
    </aside>
  );
}
