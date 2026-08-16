import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../../api'
import { queryKeys } from '../../queryKeys'
import type { BoxPatch, CropItem } from '../../types'
import './CropCard.css'

export function CropCard({ item }: { item: CropItem }) {
  const queryClient = useQueryClient()
  const [character, setCharacter] = useState(item.character)

  const mutation = useMutation({
    mutationFn: (patch: BoxPatch) => api.patchBox(item.volume, item.page, item.box_id, patch),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['crops'] })
      queryClient.invalidateQueries({ queryKey: queryKeys.pages(item.volume) })
      queryClient.invalidateQueries({ queryKey: queryKeys.volumes })
      queryClient.invalidateQueries({ queryKey: queryKeys.page(item.volume, item.page) })
    },
  })

  return (
    <div className={`crop-card ${item.selection_flag === 1 ? 'confirmed' : 'unconfirmed'}`}>
      <img src={item.crop_url} alt={item.character} loading="lazy" />
      <input
        className="nom crop-char-input"
        value={character}
        onChange={(e) => setCharacter(e.target.value)}
        onBlur={() => {
          const trimmed = character.trim()
          if (trimmed !== item.character) mutation.mutate({ character: trimmed })
        }}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
      />
      <div className="crop-card-actions">
        <button
          className={item.selection_flag === 1 ? 'active confirm' : 'confirm'}
          onClick={() => mutation.mutate({ selection_flag: 1 })}
          title="Confirm"
        >
          ✓
        </button>
        <button
          className={item.selection_flag === 0 ? 'active reject' : 'reject'}
          onClick={() => mutation.mutate({ selection_flag: 0 })}
          title="Mark unconfirmed"
        >
          ✗
        </button>
        <Link
          to={`/page/${encodeURIComponent(item.volume)}/${encodeURIComponent(item.page)}?highlight=${item.box_id}`}
          className="jump-link"
          title="Jump to page"
        >
          ↗
        </Link>
      </div>
    </div>
  )
}
