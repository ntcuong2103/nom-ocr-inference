import { useEffect, useMemo, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../../api'
import { queryKeys } from '../../queryKeys'
import type { BoxPatch, PageDetail } from '../../types'
import { PageImageCanvas } from './PageImageCanvas'
import { BoxEditPanel } from './BoxEditPanel'
import { useReviewShortcuts } from './useReviewShortcuts'
import './PageAnnotationView.css'

export function PageAnnotationView() {
  const { volume, page } = useParams<{ volume: string; page: string }>()
  const [searchParams] = useSearchParams()
  const queryClient = useQueryClient()

  const [selectedBoxId, setSelectedBoxId] = useState<number | null>(null)
  const [focusToken, setFocusToken] = useState(0)

  const pageQuery = useQuery({
    queryKey: queryKeys.page(volume!, page!),
    queryFn: () => api.getPage(volume!, page!),
    enabled: Boolean(volume && page),
  })

  useEffect(() => {
    setSelectedBoxId(null)
    const highlight = searchParams.get('highlight')
    if (highlight !== null) {
      setSelectedBoxId(Number(highlight))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volume, page])

  const mutation = useMutation({
    mutationFn: (patch: BoxPatch) =>
      api.patchBox(volume!, page!, selectedBoxId!, patch),
    onSuccess: (updatedBox) => {
      queryClient.setQueryData<PageDetail>(queryKeys.page(volume!, page!), (old) =>
        old
          ? {
              ...old,
              boxes: old.boxes.map((b) => (b.box_id === updatedBox.box_id ? updatedBox : b)),
            }
          : old,
      )
      queryClient.invalidateQueries({ queryKey: queryKeys.pages(volume!) })
      queryClient.invalidateQueries({ queryKey: queryKeys.volumes })
      queryClient.invalidateQueries({ queryKey: ['crops'] })
    },
  })

  const boxIds = useMemo(
    () => (pageQuery.data ? [...pageQuery.data.boxes].sort((a, b) => a.box_id - b.box_id).map((b) => b.box_id) : []),
    [pageQuery.data],
  )
  const selectedBox = pageQuery.data?.boxes.find((b) => b.box_id === selectedBoxId) ?? null

  useReviewShortcuts({
    boxIds,
    selectedBoxId,
    onSelect: setSelectedBoxId,
    onConfirm: () => selectedBoxId !== null && mutation.mutate({ selection_flag: 1 }),
    onReject: () => selectedBoxId !== null && mutation.mutate({ selection_flag: 0 }),
    onFocusCharacterInput: () => setFocusToken((t) => t + 1),
  })

  if (!volume || !page) return null
  if (pageQuery.isLoading) return <p className="status">Loading page…</p>
  if (pageQuery.error || !pageQuery.data) return <p className="status error">Failed to load page.</p>

  return (
    <div className="page-annotation-view">
      <header className="page-header">
        <h2>
          {volume} / <span className="nom">{page}</span>
        </h2>
        <span className="counts">
          {pageQuery.data.boxes.filter((b) => b.selection_flag === 1).length} confirmed ·{' '}
          {pageQuery.data.boxes.filter((b) => b.selection_flag === 0).length} unconfirmed
          {pageQuery.data.edited && <span className="edited-badge">edited</span>}
        </span>
      </header>
      <div className="page-body">
        <PageImageCanvas
          page={pageQuery.data}
          selectedBoxId={selectedBoxId}
          onSelectBox={setSelectedBoxId}
        />
        <BoxEditPanel
          box={selectedBox}
          saving={mutation.isPending}
          onSave={(patch) => mutation.mutate(patch)}
          focusToken={focusToken}
        />
      </div>
    </div>
  )
}
