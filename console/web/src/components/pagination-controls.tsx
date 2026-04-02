"use client";

type PaginationControlsProps = {
  offset: number;
  pageSize: number;
  itemCount: number;
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
  itemLabel = "items",
  pageSizeOptions = [25, 50, 100],
  disabled = false,
  onPageSizeChange,
  onPrevious,
  onNext,
}: PaginationControlsProps) {
  const start = itemCount === 0 ? 0 : offset + 1;
  const end = offset + itemCount;

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-3 text-sm text-zinc-400">
        <span>
          Showing {start}-{end} {itemLabel}
        </span>
        <label className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wide text-zinc-500">Page Size</span>
          <select
            value={pageSize}
            onChange={(event) => {
              onPageSizeChange(Number(event.target.value));
            }}
            className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200"
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
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-600"
        >
          Previous
        </button>
        <button
          type="button"
          onClick={onNext}
          disabled={disabled || itemCount < pageSize}
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:border-zinc-500 hover:text-white disabled:cursor-not-allowed disabled:border-zinc-800 disabled:text-zinc-600"
        >
          Next
        </button>
      </div>
    </div>
  );
}

export default PaginationControls;
