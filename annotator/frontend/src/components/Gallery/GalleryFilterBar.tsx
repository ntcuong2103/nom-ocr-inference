import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../api'
import type { VolumeSummary } from '../../types'
import './GalleryFilterBar.css'

export interface GalleryFilters {
  selection_flag?: 0 | 1
  character?: string
  volume?: string
}

export function GalleryFilterBar({
  filters,
  onChange,
  limit,
  onLimitChange,
  volumes,
}: {
  filters: GalleryFilters
  onChange: (filters: GalleryFilters) => void
  limit: number
  onLimitChange: (limit: number) => void
  volumes: VolumeSummary[]
}) {
  const [charInput, setCharInput] = useState(filters.character ?? '')
  const charsQuery = useQuery({
    queryKey: ['characters', charInput],
    queryFn: () => api.listCharacters(charInput, 20),
  })

  return (
    <div className="gallery-filter-bar">
      <label>
        Status
        <select
          value={filters.selection_flag ?? 'any'}
          onChange={(e) =>
            onChange({
              ...filters,
              selection_flag: e.target.value === 'any' ? undefined : (Number(e.target.value) as 0 | 1),
            })
          }
        >
          <option value="any">Any</option>
          <option value="1">Confirmed</option>
          <option value="0">Unconfirmed</option>
        </select>
      </label>

      <label>
        Character
        <input
          className="nom"
          list="char-suggestions"
          value={charInput}
          placeholder="e.g. 莫"
          onChange={(e) => {
            setCharInput(e.target.value)
            onChange({ ...filters, character: e.target.value || undefined })
          }}
        />
        <datalist id="char-suggestions">
          {charsQuery.data?.map((c) => (
            <option key={c.character} value={c.character}>
              {c.character || '(blank)'} — {c.count}
            </option>
          ))}
        </datalist>
      </label>

      <label>
        Volume
        <select
          value={filters.volume ?? 'any'}
          onChange={(e) =>
            onChange({ ...filters, volume: e.target.value === 'any' ? undefined : e.target.value })
          }
        >
          <option value="any">All volumes</option>
          {volumes.map((v) => (
            <option key={v.volume} value={v.volume}>
              {v.volume}
            </option>
          ))}
        </select>
      </label>

      <label>
        Per page
        <select value={limit} onChange={(e) => onLimitChange(Number(e.target.value))}>
          {[20, 50, 100, 200].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>
    </div>
  )
}
