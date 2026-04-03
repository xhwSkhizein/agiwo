"use client";

type PaginationControlsProps = {
  offset: number;
  pageSize: number;
  itemCount: number;
  totalCount?: number | null;
  hasMore?: boolean;
  itemLabel?: string;
  pageSizeOptions?: number[];
  disabled?: boolean;
  onPageSizeChange: (pageSize: number) => void;
  onPrevious: () => void;
  onNext: () => void;
};

export function PaginationControls({
  offset,
  pageSize,
  itemCount,
  totalCount = null,
  hasMore,
  itemLabel = "items",
  pageSizeOptions = [25, 50, 100],
  disabled = false,
  onPageSizeChange,
  onPrevious,
  onNext,
}: PaginationControlsProps) {
  const start = itemCount === 0 ? 0 : offset + 1;
  const end = offset + itemCount;
  const resolvedHasMore = hasMore ?? itemCount >= pageSize;
  const summaryText =
    totalCount !== null
      ? `Showing ${start}-${end} of ${totalCount} ${itemLabel}`
      : `Showing ${start}-${end} ${itemLabel}`;

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-line bg-panel px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-3 text-sm text-ink-muted">
        <span>
          {summaryText}
        </span>
        <label className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-[0.16em] text-ink-faint">Page Size</span>
          <select
            value={pageSize}
            onChange={(event) => {
              onPageSizeChange(Number(event.target.value));
            }}
            className="ui-input w-auto min-w-[5rem] px-2 py-1"
            disabled={disabled}
          >
            {pageSizeOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onPrevious}
          disabled={disabled || offset === 0}
          className="ui-button ui-button-secondary min-h-10 px-3 py-1.5 text-sm"
        >
          Previous
        </button>
        <button
          type="button"
          onClick={onNext}
          disabled={disabled || !resolvedHasMore}
          className="ui-button ui-button-secondary min-h-10 px-3 py-1.5 text-sm"
        >
          Next
        </button>
      </div>
    </div>
  );
}

export default PaginationControls;
