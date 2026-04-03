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

/**
 * Renders pagination controls: a current-range summary, a page-size selector, and Previous/Next buttons.
 *
 * @param offset - Zero-based index of the first item in the current page.
 * @param pageSize - Number of items per page currently selected.
 * @param itemCount - Number of items in the current page.
 * @param totalCount - Optional total number of items across all pages; `null` indicates unknown.
 * @param hasMore - Optional flag indicating whether there are more pages; when omitted the component infers availability from `itemCount` and `pageSize`.
 * @param itemLabel - Label used for the items in the summary (default: `"items"`).
 * @param pageSizeOptions - Options shown in the page-size selector (default: `[25, 50, 100]`).
 * @param disabled - When `true`, disables the selector and navigation buttons (default: `false`).
 * @param onPageSizeChange - Handler called with the new page size when the selector value changes.
 * @param onPrevious - Handler called when the "Previous" button is clicked.
 * @param onNext - Handler called when the "Next" button is clicked.
 * @returns A React element that displays pagination information and controls.
 */
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
