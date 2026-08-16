import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { api } from '../../api'
import { queryKeys } from '../../queryKeys'
import './PageNavSidebar.css'

function VolumeRow({ volume, expanded, onToggle }: {
  volume: { volume: string; num_pages: number; num_unconfirmed: number }
  expanded: boolean
  onToggle: () => void
}) {
  const { volume: activeVolume, page: activePage } = useParams()
  const pagesQuery = useQuery({
    queryKey: queryKeys.pages(volume.volume),
    queryFn: () => api.listPages(volume.volume),
    enabled: expanded,
  })

  return (
    <li>
      <button className="volume-toggle" onClick={onToggle}>
        <span className="chevron">{expanded ? '▾' : '▸'}</span>
        <span className="volume-name">{volume.volume}</span>
        <span className="badge">{volume.num_pages}</span>
      </button>
      {expanded && (
        <ul className="page-list">
          {pagesQuery.isLoading && <li className="muted">Loading…</li>}
          {pagesQuery.data?.map((p) => (
            <li key={p.page}>
              <Link
                to={`/page/${encodeURIComponent(volume.volume)}/${encodeURIComponent(p.page)}`}
                className={
                  activeVolume === volume.volume && activePage === p.page ? 'page-link active' : 'page-link'
                }
              >
                <span className="nom page-name">{p.page}</span>
                {p.num_unconfirmed > 0 && (
                  <span className="badge unconfirmed">{p.num_unconfirmed}</span>
                )}
                {p.edited && <span className="dot" title="has edits" />}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </li>
  )
}

export function PageNavSidebar() {
  const volumesQuery = useQuery({ queryKey: queryKeys.volumes, queryFn: api.listVolumes })
  const [expandedVolume, setExpandedVolume] = useState<string | null>(null)

  return (
    <div className="page-nav-sidebar">
      {volumesQuery.isLoading && <p className="muted">Loading volumes…</p>}
      {volumesQuery.error && <p className="error">Failed to load volumes</p>}
      <ul className="volume-list">
        {volumesQuery.data?.map((v) => (
          <VolumeRow
            key={v.volume}
            volume={v}
            expanded={expandedVolume === v.volume}
            onToggle={() => setExpandedVolume(expandedVolume === v.volume ? null : v.volume)}
          />
        ))}
      </ul>
    </div>
  )
}
