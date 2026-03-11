export function formatLocalDateTime(
  value: string | null | undefined,
  fallback = "-"
): string {
  if (!value) {
    return fallback;
  }

  return new Date(value).toLocaleString();
}

export function formatRoundedMs(
  value: number | null | undefined,
  fallback = "-"
): string {
  if (value === null || value === undefined || value <= 0) {
    return fallback;
  }

  return `${Math.round(value)}ms`;
}
