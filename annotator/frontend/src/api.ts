import type {
  Box,
  BoxPatch,
  CharacterCount,
  CropsFilter,
  CropsPage,
  PageDetail,
  PageSummary,
  VolumeSummary,
} from './types'

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  listVolumes: () => fetchJson<VolumeSummary[]>('/api/volumes'),

  listPages: (volume: string) =>
    fetchJson<PageSummary[]>(`/api/volumes/${encodeURIComponent(volume)}/pages`),

  getPage: (volume: string, page: string) =>
    fetchJson<PageDetail>(
      `/api/pages/${encodeURIComponent(volume)}/${encodeURIComponent(page)}`,
    ),

  patchBox: (volume: string, page: string, boxId: number, patch: BoxPatch) =>
    fetchJson<Box>(
      `/api/pages/${encodeURIComponent(volume)}/${encodeURIComponent(page)}/boxes/${boxId}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      },
    ),

  listCrops: (filter: CropsFilter) => {
    const params = new URLSearchParams()
    if (filter.selection_flag !== undefined)
      params.set('selection_flag', String(filter.selection_flag))
    if (filter.character !== undefined) params.set('character', filter.character)
    if (filter.volume !== undefined) params.set('volume', filter.volume)
    if (filter.page !== undefined) params.set('page', filter.page)
    params.set('limit', String(filter.limit))
    params.set('offset', String(filter.offset))
    return fetchJson<CropsPage>(`/api/crops?${params.toString()}`)
  },

  listCharacters: (q: string, limit = 50) => {
    const params = new URLSearchParams()
    if (q) params.set('q', q)
    params.set('limit', String(limit))
    return fetchJson<CharacterCount[]>(`/api/characters?${params.toString()}`)
  },
}
