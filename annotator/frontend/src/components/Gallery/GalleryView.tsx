import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../api'
import { queryKeys } from '../../queryKeys'
import { GalleryFilterBar, type GalleryFilters } from './GalleryFilterBar'
import { CropGrid } from './CropGrid'
import { Pagination } from './Pagination'
import './GalleryView.css'

export function GalleryView() {
  const [filters, setFilters] = useState<GalleryFilters>({})
  const [limit, setLimit] = useState(50)
  const [offset, setOffset] = useState(0)

  const volumesQuery = useQuery({ queryKey: queryKeys.volumes, queryFn: api.listVolumes })
  const cropsQuery = useQuery({
    queryKey: queryKeys.crops({ ...filters, limit, offset }),
    queryFn: () => api.listCrops({ ...filters, limit, offset }),
  })

  return (
    <div className="gallery-view">
      <GalleryFilterBar
        filters={filters}
        onChange={(f) => {
          setFilters(f)
          setOffset(0)
        }}
        limit={limit}
        onLimitChange={(l) => {
          setLimit(l)
          setOffset(0)
        }}
        volumes={volumesQuery.data ?? []}
      />
      {cropsQuery.isLoading && <p className="status">Loading crops…</p>}
      {cropsQuery.error && <p className="status error">Failed to load crops.</p>}
      {cropsQuery.data && (
        <>
          <CropGrid items={cropsQuery.data.items} />
          <Pagination
            total={cropsQuery.data.total}
            limit={limit}
            offset={offset}
            onOffsetChange={setOffset}
          />
        </>
      )}
    </div>
  )
}
