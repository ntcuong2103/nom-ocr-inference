import type { CropsFilter } from './types'

export const queryKeys = {
  volumes: ['volumes'] as const,
  pages: (volume: string) => ['pages', volume] as const,
  page: (volume: string, page: string) => ['page', volume, page] as const,
  crops: (filter: CropsFilter) => ['crops', filter] as const,
  characters: (q: string) => ['characters', q] as const,
}
