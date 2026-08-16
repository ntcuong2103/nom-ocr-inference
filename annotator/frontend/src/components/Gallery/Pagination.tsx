import './Pagination.css'

export function Pagination({
  total,
  limit,
  offset,
  onOffsetChange,
}: {
  total: number
  limit: number
  offset: number
  onOffsetChange: (offset: number) => void
}) {
  const page = Math.floor(offset / limit) + 1
  const pageCount = Math.max(1, Math.ceil(total / limit))

  return (
    <div className="pagination">
      <button disabled={offset === 0} onClick={() => onOffsetChange(Math.max(0, offset - limit))}>
        ← Prev
      </button>
      <span>
        Page {page} / {pageCount} ({total} total)
      </span>
      <button
        disabled={offset + limit >= total}
        onClick={() => onOffsetChange(offset + limit)}
      >
        Next →
      </button>
    </div>
  )
}
